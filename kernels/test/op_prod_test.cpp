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
using exec_aten::optional;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor&
op_prod_out(const Tensor& self, optional<ScalarType> dtype, Tensor& out) {
  executorch::runtime::KernelRuntimeContext context{};
  return torch::executor::aten::prod_outf(context, self, dtype, out);
}

Tensor& op_prod_int_out(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& out) {
  executorch::runtime::KernelRuntimeContext context{};
  return torch::executor::aten::prod_outf(
      context, self, dim, keepdim, dtype, out);
}

class OpProdOutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    torch::executor::runtime_init();
  }
};

class OpProdIntOutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    torch::executor::runtime_init();
  }
};

TEST_F(OpProdOutTest, SmokeTest) {
  TensorFactory<ScalarType::Float> tfFloat;

  Tensor self = tfFloat.make({2, 3}, {1, 2, 3, 4, 5, 6});
  optional<ScalarType> dtype{};
  Tensor out = tfFloat.zeros({});
  Tensor out_expected = tfFloat.make({}, {720});
  op_prod_out(self, dtype, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST_F(OpProdIntOutTest, SmokeTest) {
  TensorFactory<ScalarType::Float> tfFloat;

  Tensor self = tfFloat.make({2, 3}, {1, 2, 3, 4, 5, 6});
  int64_t dim = 0;
  bool keepdim = false;
  optional<ScalarType> dtype{};
  Tensor out = tfFloat.zeros({3});
  Tensor out_expected = tfFloat.make({3}, {4, 10, 18});
  op_prod_int_out(self, dim, keepdim, dtype, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}

TEST_F(OpProdIntOutTest, SmokeTestKeepdim) {
  TensorFactory<ScalarType::Float> tfFloat;

  Tensor self = tfFloat.make({2, 3}, {1, 2, 3, 4, 5, 6});
  int64_t dim = 0;
  bool keepdim = true;
  optional<ScalarType> dtype{};
  Tensor out = tfFloat.zeros({1, 3});
  Tensor out_expected = tfFloat.make({1, 3}, {4, 10, 18});
  op_prod_int_out(self, dim, keepdim, dtype, out);
  EXPECT_TENSOR_CLOSE(out, out_expected);
}
