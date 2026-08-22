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
using cortex_m::native::quantize_per_tensor_out;

void test_dtype_int8() {
  TensorFactory<ScalarType::Float> tf;

  Tensor input = tf.full({3, 5}, 4);
  double scale = 0.5;

  int64_t zero_point = 108;
  int64_t quant_min = 0;
  int64_t quant_max = 127;

  TensorFactory<ScalarType::Char> tfo;
  Tensor out = tfo.zeros({3, 5});
  // 4 / 0.5 + 108 = 116
  Tensor expected = tfo.full({3, 5}, 116);

  KernelRuntimeContext ctx;
  quantize_per_tensor_out(
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

void test_dtype_int16() {
  TensorFactory<ScalarType::Float> tf;

  // Shape sized to exercise the MVE int16 path (8-element chunks).
  Tensor input = tf.full({4, 4}, 4);
  double scale = 0.5;

  int64_t zero_point = 1000;
  int64_t quant_min = -32768;
  int64_t quant_max = 32767;

  TensorFactory<ScalarType::Short> tfo;
  Tensor out = tfo.zeros({4, 4});
  // 4 / 0.5 + 1000 = 1008
  Tensor expected = tfo.full({4, 4}, 1008);

  KernelRuntimeContext ctx;
  quantize_per_tensor_out(
      ctx,
      input,
      scale,
      zero_point,
      quant_min,
      quant_max,
      ScalarType::Short,
      out);

  EXPECT_TENSOR_EQ(out, expected);
}

void test_dtype_int16_clamp() {
  TensorFactory<ScalarType::Float> tf;

  // Input large enough to clamp at quant_max after scaling.
  Tensor input = tf.full({4, 4}, 100000);
  double scale = 0.5;

  int64_t zero_point = 0;
  int64_t quant_min = -32768;
  int64_t quant_max = 32767;

  TensorFactory<ScalarType::Short> tfo;
  Tensor out = tfo.zeros({4, 4});
  Tensor expected = tfo.full({4, 4}, 32767);

  KernelRuntimeContext ctx;
  quantize_per_tensor_out(
      ctx,
      input,
      scale,
      zero_point,
      quant_min,
      quant_max,
      ScalarType::Short,
      out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST(OpQuantizeOutTest, AllDtypesSupported) {
  test_dtype_int8();
  test_dtype_int16();
  test_dtype_int16_clamp();
}
