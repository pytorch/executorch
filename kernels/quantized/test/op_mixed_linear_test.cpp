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
using exec_aten::optional;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using executorch::runtime::KernelRuntimeContext;
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

  Tensor input = tf.make(
      /*sizes=*/{1, 3},
      /*data=*/{1.0, 1.5, 2.0});
  Tensor weight = tf_char.make(
      /*sizes=*/{2, 3},
      /*data=*/{5, 3, 1, 4, 2, 1});
  Tensor weight_scales = tf.make(
      /*sizes=*/{2},
      /*data=*/{0.2, 0.4});
  const optional<Tensor> opt_weight_zp{};
  const optional<ScalarType> opt_dtype_out{};

  Tensor out = tf_out.zeros({1, 2});

  Tensor expected = tf_out.make(
      /*sizes=*/{1, 2},
      /*data=*/{2.3, 3.6});

  KernelRuntimeContext ctx{};

  quantized_mixed_linear_out(
      ctx, input, weight, weight_scales, opt_weight_zp, opt_dtype_out, out);

  EXPECT_TENSOR_CLOSE(out, expected);
}

TEST_F(OpQuantizedMixedDtypeLinearTest, FloatInputFloatOutput) {
  test_dtype<ScalarType::Float, ScalarType::Float>();
}

#if 0
// need <<
TEST_F(OpQuantizedMixedDtypeLinearTest, FloatInputHalfOutput) {
  test_dtype<ScalarType::Float, ScalarType::Half>();
}

// need to relax tolerance
TEST_F(OpQuantizedMixedDtypeLinearTest, HalfInputFloatOutput) {
  test_dtype<ScalarType::Half, ScalarType::Float>();
}

// need <<
TEST_F(OpQuantizedMixedDtypeLinearTest, HalfInputHalfOutput) {
  test_dtype<ScalarType::Half, ScalarType::Half>();
}
#endif

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
