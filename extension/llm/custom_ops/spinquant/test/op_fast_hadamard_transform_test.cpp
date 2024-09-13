/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/custom_ops/op_fast_hadamard_transform.h>
#include <executorch/extension/llm/custom_ops/spinquant/third-party/FFHT/dumb_fht.h>
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>

#include <gtest/gtest.h>

#include <cmath>
#include <random>

using exec_aten::Tensor;

namespace {
Tensor& fast_hadamard_transform_nocontext(const Tensor& vec, Tensor& out) {
  exec_aten::RuntimeContext context;
  return torch::executor::native::fast_hadamard_transform_out(
      context, vec, out);
}

void reference_fht_impl(float* buf, int n) {
  dumb_fht(buf, std::log2<int>(n));
  const auto root_n = std::sqrt(n);
  for (int ii = 0; ii < n; ++ii) {
    buf[ii] /= root_n;
  }
}
} // namespace

TEST(FastHadamardTransformTest, EmptyInput) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  auto vec = tfFloat.zeros({0});
  auto out = tfFloat.zeros({0});
  auto result = fast_hadamard_transform_nocontext(vec, out);
  EXPECT_EQ(result.numel(), 0);
}

TEST(FastHadamardTransformTest, SingleElementInput) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  auto vec = tfFloat.ones({1});
  auto out = tfFloat.zeros({1});
  auto result = fast_hadamard_transform_nocontext(vec, out);
  EXPECT_EQ(result.numel(), 1);
  // FHT of a single element is a no-op.
  EXPECT_EQ(result.const_data_ptr<float>()[0], 1);
}

TEST(FastHadamardTransformTest, FourKInput) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist;
  std::vector<float> data(4096);
  for (int ii = 0; ii < data.size(); ++ii) {
    data[ii] = dist(gen);
  }
  auto vec = tfFloat.make({4096}, data);
  auto out = tfFloat.zeros({4096});
  auto result = fast_hadamard_transform_nocontext(vec, out);

  std::vector<float> reference_result = data;
  reference_fht_impl(reference_result.data(), reference_result.size());

  const float* const result_data = result.const_data_ptr<float>();
  for (int ii = 0; ii < 4096; ++ii) {
    EXPECT_FLOAT_EQ(result_data[ii], reference_result[ii]);
  }
}

TEST(FastHadamardTransformTest, MultipleRows) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist;
  std::vector<float> data(8 * 8 * 8);
  for (int ii = 0; ii < data.size(); ++ii) {
    data[ii] = dist(gen);
  }
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
