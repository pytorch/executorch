/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/llama2/custom_ops/op_randomized_fast_hadamard_transform.h>

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <gtest/gtest.h>

#include <cmath>
#include <random>

using exec_aten::Tensor;

namespace {
Tensor& randomized_fast_hadamard_transform_nocontext(
    const Tensor& vec,
    const Tensor& randomization_bitvec,
    Tensor& out) {
  exec_aten::RuntimeContext context;
  return torch::executor::native::randomized_fast_hadamard_transform_out(
      context, vec, randomization_bitvec, out);
}

// The following "dumb_fht" function is from
// https://github.com/FALCONN-LIB/FFHT/blob/master/test_float.c . It
// has the following required copyright notice:
// The MIT License (MIT)
//
// Copyright (c) 2015 Alexandr Andoni, Piotr Indyk, Thijs Laarhoven, Ilya
// Razenshteyn, Ludwig Schmidt
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
void dumb_fht(float* buf, int log_n) {
  int n = 1 << log_n;
  for (int i = 0; i < log_n; ++i) {
    int s1 = 1 << i;
    int s2 = s1 << 1;
    for (int j = 0; j < n; j += s2) {
      for (int k = 0; k < s1; ++k) {
        float u = buf[j + k];
        float v = buf[j + k + s1];
        buf[j + k] = u + v;
        buf[j + k + s1] = u - v;
      }
    }
  }
}

void reference_fht_impl(float* buf, int n) {
  dumb_fht(buf, std::log2<int>(n));
  const auto root_n = std::sqrt(n);
  for (int ii = 0; ii < n; ++ii) {
    buf[ii] /= root_n;
  }
}
} // namespace

TEST(RandomizedFastHadamardTransformTest, EmptyInput) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  auto vec = tfFloat.zeros({0});
  auto randomization_bitvec = tfByte.zeros({0});
  auto out = tfFloat.zeros({0});
  auto result = randomized_fast_hadamard_transform_nocontext(
      vec, randomization_bitvec, out);
  EXPECT_EQ(result.numel(), 0);
}

TEST(RandomizedFastHadamardTransformTest, SingleElementInput) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  auto vec = tfFloat.ones({1});
  auto randomization_bitvec = tfByte.zeros({1});
  auto out = tfFloat.zeros({1});
  auto result = randomized_fast_hadamard_transform_nocontext(
      vec, randomization_bitvec, out);
  EXPECT_EQ(result.numel(), 1);
  // FHT of a single element is a no-op.
  EXPECT_EQ(result.const_data_ptr<float>()[0], 1);

  vec.mutable_data_ptr<float>()[0] = 42;
  result = randomized_fast_hadamard_transform_nocontext(
      vec, randomization_bitvec, out);
  EXPECT_EQ(result.const_data_ptr<float>()[0], 42);

  randomization_bitvec.mutable_data_ptr<char>()[0] = 1;
  result = randomized_fast_hadamard_transform_nocontext(
      vec, randomization_bitvec, out);
  EXPECT_EQ(result.const_data_ptr<float>()[0], -42);
}

TEST(RandomizedFastHadamardTransformTest, FourKInput) {
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Float> tfFloat;
  torch::executor::testing::TensorFactory<exec_aten::ScalarType::Byte> tfByte;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist;
  std::vector<float> data(4096);
  for (int ii = 0; ii < data.size(); ++ii) {
    data[ii] = dist(gen);
  }
  auto vec = tfFloat.make({4096}, data);
  auto randomization_bitvec = tfByte.full({4096 / 8}, 0xA5);
  auto out = tfFloat.zeros({4096});
  auto result = randomized_fast_hadamard_transform_nocontext(
      vec, randomization_bitvec, out);

  std::vector<float> reference_result = data;
  reference_fht_impl(reference_result.data(), reference_result.size());

  const float* const result_data = result.const_data_ptr<float>();
  for (int ii = 0; ii < 4096; ++ii) {
    const bool should_negate = (ii % 2) != (ii % 8 < 4);
    EXPECT_FLOAT_EQ(
        should_negate ? -result_data[ii] : result_data[ii],
        reference_result[ii]);
  }
}
