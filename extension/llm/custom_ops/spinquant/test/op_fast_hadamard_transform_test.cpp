/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/custom_ops/op_fast_hadamard_transform.h>
#include <executorch/extension/llm/custom_ops/spinquant/test/fast_hadamard_transform_test_impl.h>
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>

#include <gtest/gtest.h>

#include <cmath>

using exec_aten::Tensor;

using executorch::runtime::testing::fast_hadamard_transform_28N_with_transpose;
using executorch::runtime::testing::random_floats;
using executorch::runtime::testing::reference_fht_impl;

namespace {
Tensor& fast_hadamard_transform_nocontext(const Tensor& vec, Tensor& out) {
  exec_aten::RuntimeContext context;
  return torch::executor::native::fast_hadamard_transform_out(
      context, vec, out);
}
} // namespace

TEST(OpFastHadamardTransformTest, EmptyInput) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  auto vec = tfFloat.zeros({0});
  auto out = tfFloat.zeros({0});
  auto result = fast_hadamard_transform_nocontext(vec, out);
  EXPECT_EQ(result.numel(), 0);
}

TEST(OpFastHadamardTransformTest, SingleElementInput) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  auto vec = tfFloat.ones({1});
  auto out = tfFloat.zeros({1});
  auto result = fast_hadamard_transform_nocontext(vec, out);
  EXPECT_EQ(result.numel(), 1);
  // FHT of a single element is a no-op.
  EXPECT_EQ(result.const_data_ptr<float>()[0], 1);
}

TEST(OpFastHadamardTransformTest, FourKInput) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  std::vector<float> data = random_floats(4096);
  auto vec = tfFloat.make({4096}, data);
  auto out = tfFloat.zeros({4096});
  auto result = fast_hadamard_transform_nocontext(vec, out);

  std::vector<float> reference_result = data;
  reference_fht_impl(reference_result.data(), reference_result.size());

  const float* const result_data = result.const_data_ptr<float>();
  for (int ii = 0; ii < data.size(); ++ii) {
    EXPECT_FLOAT_EQ(result_data[ii], reference_result[ii]);
  }
}

TEST(OpFastHadamardTransformTest, MultipleRows) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  std::vector<float> data = random_floats(8 * 8 * 8);
  auto mat = tfFloat.make({8, 8, 8}, data);
  auto out = tfFloat.zeros({8, 8, 8});

  auto result = fast_hadamard_transform_nocontext(mat, out);

  std::vector<float> reference_result = data;
  for (int ii = 0; ii < 8; ++ii) {
    for (int jj = 0; jj < 8; ++jj) {
      reference_fht_impl(&reference_result[ii * 64 + jj * 8], 8);
    }
  }

  const float* const result_data = result.const_data_ptr<float>();
  for (int ii = 0; ii < data.size(); ++ii) {
    EXPECT_FLOAT_EQ(result_data[ii], reference_result[ii]);
  }
}

TEST(OpFastHadamardTransformTest, Basic28N) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  constexpr int kTestLogSize = 7;
  constexpr int kTestPowerOfTwoSize = 1 << kTestLogSize;
  constexpr int kTestTotalSize = kTestPowerOfTwoSize * 28;
  std::vector<float> data = random_floats(kTestTotalSize);
  auto vec = tfFloat.make({kTestTotalSize}, data);
  auto out = tfFloat.zeros({kTestTotalSize});

  // The operator is supposed to autodetect 28 * 2**N size and handle
  // accordingly.
  auto result = fast_hadamard_transform_nocontext(vec, out);

  std::vector<float> reference_result = data;
  fast_hadamard_transform_28N_with_transpose(
      reference_result.data(), kTestLogSize);

  const float* const result_data = result.const_data_ptr<float>();
  for (int ii = 0; ii < data.size(); ++ii) {
    EXPECT_FLOAT_EQ(result_data[ii], reference_result[ii]);
  }
}

TEST(OpFastHadamardTransformTest, InvalidSize) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  auto mat = tfFloat.zeros({3});
  auto out = tfFloat.zeros({3});

  exec_aten::RuntimeContext context;
  torch::executor::native::fast_hadamard_transform_out(context, mat, out);
  EXPECT_NE(context.failure_state(), executorch::runtime::Error::Ok);
}
