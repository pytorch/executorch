/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/NativeFunctions.h> // Declares the aten operator
#include <executorch/kernels/quantized/NativeFunctions.h> // Declares the quantized operator
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/runtime.h>

#include <gtest/gtest.h>

using namespace ::testing;
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::ET_RUNTIME_NAMESPACE::KernelRuntimeContext;
using std::optional;
using torch::executor::native::quantized_mixed_mm_out;
using torch::executor::testing::TensorFactory;

class OpQuantizedMixedMMTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    torch::executor::runtime_init();
  }
};

template <ScalarType DTYPE>
void test_dtype() {
  TensorFactory<DTYPE> tf;
  TensorFactory<ScalarType::Char> tf_char;

  Tensor input = tf.make(
      /*sizes=*/{1, 3},
      /*data=*/{1.0, 1.5, 2.0});
  Tensor weight = tf_char.make(
      /*sizes=*/{3, 2},
      /*data=*/{5, 4, 3, 2, 1, 1});
  Tensor weight_scales = tf.make(
      /*sizes=*/{3},
      /*data=*/{0.25, 0.5, 0.125});
  const optional<Tensor> opt_weight_zp{};

  Tensor out = tf.zeros({1, 2});

  Tensor expected = tf.make(
      /*sizes=*/{1, 2},
      /*data=*/{3.75, 2.75});

  KernelRuntimeContext ctx{};

  quantized_mixed_mm_out(ctx, input, weight, weight_scales, opt_weight_zp, out);

  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST_F(OpQuantizedMixedMMTest, FloatInput) {
  test_dtype<ScalarType::Float>();
}

TEST_F(OpQuantizedMixedMMTest, HalfInput) {
  test_dtype<ScalarType::Half>();
}

TEST_F(OpQuantizedMixedMMTest, HalfComputesInFloat) {
  TensorFactory<ScalarType::Half> tf;
  TensorFactory<ScalarType::Char> tf_char;
  KernelRuntimeContext ctx{};

  // Each output sums 4096 products of 1. A Half running sum stops growing at
  // 2048, so without float accumulation the result is 2048 instead of 4096.
  constexpr int K = 4096;
  Tensor out = tf.zeros({1, 2});
  quantized_mixed_mm_out(
      ctx, tf.ones({1, K}), tf_char.ones({K, 2}), tf.ones({K}), {}, out);
  EXPECT_TENSOR_EQ(out, tf.full({1, 2}, K));

  // 600 * 127 overflows Half, while the scaled product, about 762, fits.
  Tensor small_out = tf.zeros({1, 1});
  quantized_mixed_mm_out(
      ctx,
      tf.full({1, 1}, 600),
      tf_char.full({1, 1}, 127),
      tf.full({1}, 0.01),
      {},
      small_out);
  EXPECT_TENSOR_CLOSE(small_out, tf.full({1, 1}, 762));
}
