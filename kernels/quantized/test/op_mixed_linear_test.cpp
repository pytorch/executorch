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
using torch::executor::native::quantized_mixed_linear_out;
using torch::executor::testing::TensorFactory;

class OpQuantizedMixedDtypeLinearTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Since these tests cause ET_LOG to be called, the PAL must be initialized
    // first.
    torch::executor::runtime_init();
  }
};

template <ScalarType DTYPE, ScalarType DTYPE_OUT>
void test_dtype() {
  TensorFactory<DTYPE> tf;
  TensorFactory<ScalarType::Char> tf_char;
  TensorFactory<DTYPE_OUT> tf_out;

  // Every value here, and the result, is exact in Half.
  Tensor input = tf.make(
      /*sizes=*/{1, 3},
      /*data=*/{1.0, 1.5, 2.0});
  Tensor weight = tf_char.make(
      /*sizes=*/{2, 3},
      /*data=*/{5, 3, 1, 4, 2, 1});
  Tensor weight_scales = tf.make(
      /*sizes=*/{2},
      /*data=*/{0.25, 0.5});
  const optional<Tensor> opt_weight_zp{};
  const optional<ScalarType> opt_dtype_out{};

  Tensor out = tf_out.zeros({1, 2});

  Tensor expected = tf_out.make(
      /*sizes=*/{1, 2},
      /*data=*/{2.875, 4.5});

  KernelRuntimeContext ctx{};

  quantized_mixed_linear_out(
      ctx, input, weight, weight_scales, opt_weight_zp, opt_dtype_out, out);

  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST_F(OpQuantizedMixedDtypeLinearTest, FloatInputFloatOutput) {
  test_dtype<ScalarType::Float, ScalarType::Float>();
}

TEST_F(OpQuantizedMixedDtypeLinearTest, HalfComputesInFloat) {
  TensorFactory<ScalarType::Half> tf;
  TensorFactory<ScalarType::Char> tf_char;
  KernelRuntimeContext ctx{};

  // Each output sums 4096 products of 1. A Half running sum stops growing at
  // 2048, so without float accumulation the result is 2048 instead of 4096.
  // One scale per row makes one group of 4096 products, and one scale per
  // element makes 4096 groups of one product.
  constexpr int K = 4096;
  Tensor in = tf.ones({1, K});
  Tensor weight = tf_char.ones({2, K});
  Tensor out = tf.zeros({1, 2});
  quantized_mixed_linear_out(ctx, in, weight, tf.ones({2}), {}, {}, out);
  EXPECT_TENSOR_EQ(out, tf.full({1, 2}, K));
  Tensor grouped_out = tf.zeros({1, 2});
  quantized_mixed_linear_out(
      ctx, in, weight, tf.ones({2, K}), {}, {}, grouped_out);
  EXPECT_TENSOR_EQ(grouped_out, tf.full({1, 2}, K));

  // 600 * 127 overflows Half, while the scaled product, about 762, fits.
  Tensor small_out = tf.zeros({1, 1});
  quantized_mixed_linear_out(
      ctx,
      tf.full({1, 1}, 600),
      tf_char.full({1, 1}, 127),
      tf.full({1}, 0.01),
      {},
      {},
      small_out);
  EXPECT_TENSOR_CLOSE(small_out, tf.full({1, 1}, 762));
}

TEST_F(OpQuantizedMixedDtypeLinearTest, FloatInputDoubleOutput) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Char> tf_char;
  TensorFactory<ScalarType::Double> tf_double;
  KernelRuntimeContext ctx{};

  // (1 + 2^-23) * 127 rounds to 127 + 2^-16 in float, but not in double.
  Tensor out = tf_double.zeros({1, 1});
  quantized_mixed_linear_out(
      ctx,
      tf.full({1, 1}, 1.0f + 0x1p-23f),
      tf_char.full({1, 1}, 127),
      tf.ones({1}),
      {},
      {},
      out);
  EXPECT_TENSOR_EQ(out, tf_double.full({1, 1}, 127.0 + 0x1p-16));
}

TEST_F(OpQuantizedMixedDtypeLinearTest, FloatInputHalfOutput) {
  test_dtype<ScalarType::Float, ScalarType::Half>();
}

TEST_F(OpQuantizedMixedDtypeLinearTest, HalfInputFloatOutput) {
  test_dtype<ScalarType::Half, ScalarType::Float>();
}

TEST_F(OpQuantizedMixedDtypeLinearTest, HalfInputHalfOutput) {
  test_dtype<ScalarType::Half, ScalarType::Half>();
}

template <ScalarType DTYPE, ScalarType DTYPE_OUT>
void test_dtype_partials() {
  TensorFactory<DTYPE> tf;
  TensorFactory<ScalarType::Char> tf_char;
  TensorFactory<DTYPE_OUT> tf_out;

  Tensor input = tf.make(
      /*sizes=*/{1, 3},
      /*data=*/{1.0, 1.5, 2.0});
  Tensor weight = tf_char.make(
      /*sizes=*/{2, 3},
      /*data=*/{5, 3, 1, 4, 2, 1});
  Tensor weight_scales = tf.make(
      /*sizes=*/{2, 2},
      /*data=*/{0.2, 1, 0.4, 0.5});
  const optional<Tensor> opt_weight_zp{};
  const optional<ScalarType> opt_dtype_out{};

  Tensor out = tf_out.zeros({1, 2});

  Tensor expected = tf_out.make(
      /*sizes=*/{1, 2},
      /*data=*/
      {(1.0 * 5 + 1.5 * 3) * 0.2 + 2.0 * 1 * 1,
       (1.0 * 4 + 1.5 * 2) * 0.4 + 2.0 * 1 * 0.5});

  KernelRuntimeContext ctx{};

  quantized_mixed_linear_out(
      ctx, input, weight, weight_scales, opt_weight_zp, opt_dtype_out, out);

  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST_F(OpQuantizedMixedDtypeLinearTest, FloatInputFloatOutput_Partials) {
  test_dtype_partials<ScalarType::Float, ScalarType::Float>();
}

#if 0
// need <<
TEST_F(OpQuantizedMixedDtypeLinearTest, FloatInputHalfOutput_Partials) {
  test_dtype_partials<ScalarType::Float, ScalarType::Half>();
}

// need to relax tolerance
TEST_F(OpQuantizedMixedDtypeLinearTest, HalfInputFloatOutput_Partials) {
  test_dtype_partials<ScalarType::Half, ScalarType::Float>();
}

// need <<
TEST_F(OpQuantizedMixedDtypeLinearTest, HalfInputHalfOutput_Partials) {
  test_dtype_partials<ScalarType::Half, ScalarType::Half>();
}
#endif
