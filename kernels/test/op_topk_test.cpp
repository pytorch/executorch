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
using exec_aten::IntArrayRef;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

std::tuple<Tensor&, Tensor&> op_topk_values(
    const Tensor& input,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted,
    Tensor& values,
    Tensor& indices) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::topk_outf(
      context, input, k, dim, largest, sorted, values, indices);
}

class OpTopkValuesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    torch::executor::runtime_init();
  }
};

TEST_F(OpTopkValuesTest, SmokeTest) {
  TensorFactory<ScalarType::Float> tfFloat;
  TensorFactory<ScalarType::Long> tfLong;

  Tensor input =
      tfFloat.make({3, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  int64_t k = 2;
  int64_t dim = 0;
  bool largest = true;
  bool sorted = true;
  Tensor values = tfFloat.zeros({2, 2, 2});
  Tensor indices = tfLong.zeros({2, 2, 2});
  Tensor values_expected = tfFloat.make({2, 2, 2}, {9, 10, 11, 12, 5, 6, 7, 8});
  Tensor indices_expected = tfLong.make({2, 2, 2}, {2, 2, 2, 2, 1, 1, 1, 1});
  op_topk_values(input, k, dim, largest, sorted, values, indices);
  EXPECT_TENSOR_CLOSE(values, values_expected);
  EXPECT_TENSOR_EQ(indices, indices_expected);
}
