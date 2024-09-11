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
      /*data=*/{0.2, 0.4, 0.5});
  const optional<Tensor> opt_weight_zp{};

  Tensor out = tf.zeros({1, 2});

  Tensor expected = tf.make(
      /*sizes=*/{1, 2},
      /*data=*/{3.8, 3.0});

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
