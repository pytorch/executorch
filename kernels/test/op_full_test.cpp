/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/test/FunctionHeaderWrapper.h> // Declares the operator
#include <executorch/kernels/test/supported_features.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/test/utils/DeathTest.h>

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::IntArrayRef;
using exec_aten::MemoryFormat;
using exec_aten::optional;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

Tensor&
op_full_out(const IntArrayRef sizes, const Scalar& fill_value, Tensor& out) {
  exec_aten::RuntimeContext context{};
  return torch::executor::aten::full_outf(context, sizes, fill_value, out);
}

template <ScalarType DTYPE>
void test_ones_out(std::vector<int32_t>&& size_int32_t) {
  TensorFactory<DTYPE> tf;
  std::vector<int64_t> size_int64_t(size_int32_t.begin(), size_int32_t.end());
  auto aref = IntArrayRef(size_int64_t.data(), size_int64_t.size());

  // Before: `out` consists of 0s.
  Tensor out = tf.zeros(size_int32_t);

  // After: `out` consists of 1s.
  op_full_out(aref, 1, out);

  EXPECT_TENSOR_EQ(out, tf.ones(size_int32_t));
}

#define GENERATE_TEST(_, DTYPE)                  \
  TEST(OpFullOutTest, DTYPE##Tensors) {          \
    test_ones_out<ScalarType::DTYPE>({});        \
    test_ones_out<ScalarType::DTYPE>({1});       \
    test_ones_out<ScalarType::DTYPE>({1, 1, 1}); \
    test_ones_out<ScalarType::DTYPE>({2, 0, 4}); \
    test_ones_out<ScalarType::DTYPE>({2, 3, 4}); \
  }

ET_FORALL_REAL_TYPES(GENERATE_TEST)
