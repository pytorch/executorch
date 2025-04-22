/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cortex_m/ops/NativeFunctions.h> // Declares the operator
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <gtest/gtest.h>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::KernelRuntimeContext;
using torch::executor::testing::TensorFactory;

// Test op
using cortex_m::native::dequantize_per_tensor_out;

void test_dtype() {
  TensorFactory<ScalarType::Char> tf;

  Tensor input = tf.full({3, 5}, 4);
  double scale = 0.5;

  int64_t zero_point = 108;
  int64_t quant_min = -128;
  int64_t quant_max = 127;

  TensorFactory<ScalarType::Float> tfo;
  Tensor out = tfo.zeros({3, 5});
  // (4 - 108) * 0.5 = -52
  Tensor expected = tfo.full({3, 5}, -52.0);

  KernelRuntimeContext ctx;
  dequantize_per_tensor_out(
      ctx,
      input,
      scale,
      zero_point,
      quant_min,
      quant_max,
      ScalarType::Char,
      out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpDequantizeOutTest, AllDtypesSupported) {
  test_dtype();
}
