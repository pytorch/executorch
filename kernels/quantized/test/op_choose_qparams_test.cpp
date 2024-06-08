/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/quantized/NativeFunctions.h> // Declares the operator
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/test/utils/DeathTest.h>

#include <gtest/gtest.h>
#include <limits>

using namespace ::testing;
using exec_aten::ArrayRef;
using exec_aten::Scalar;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::native::choose_qparams_tensor_out;
using torch::executor::testing::TensorFactory;

/// A generic smoke test that works for any dtype that supports ones() and
/// zeros().
template <ScalarType DTYPE>
void test_dtype() {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor input = tf_float.make({2, 2}, {1.0, 2.5, 3.2, 15.4});
  Tensor scale_out = tf_double.zeros({1});
  Tensor zero_point_out = tf_long.zeros({1});
  Tensor expected_scale = tf_double.make({1}, {0.0603922});
  Tensor expected_zero_point = tf_long.make({1}, {0});

  int64_t quant_min = 0;
  int64_t quant_max = 255;

  choose_qparams_tensor_out(
      input, quant_min, quant_max, 0.0, DTYPE, scale_out, zero_point_out);

  EXPECT_TENSOR_CLOSE(scale_out, expected_scale);
  EXPECT_TENSOR_EQ(zero_point_out, expected_zero_point);
}

TEST(OpChooseQparamsTensorOutTest, AllDtypesSupported) {
  test_dtype<ScalarType::Byte>();
}
