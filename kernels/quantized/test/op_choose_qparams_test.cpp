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
using torch::executor::native::choose_qparams_per_token_asymmetric_out;
using torch::executor::native::choose_qparams_tensor_out;
using torch::executor::testing::TensorFactory;

/// A generic smoke test that works for any dtype that supports ones() and
/// zeros().
template <ScalarType DTYPE>
void test_dtype() {
  et_pal_init();
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

TEST(OpChooseQparamsPerTokenAsymmetricTensorOutTest, Float) {
  et_pal_init();
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor input = tf_float.make({2, 3}, {-0.5, 0.3, 1.2, 0.1, -0.8, 2.1});
  Tensor scale_out = tf_double.zeros({2, 1});
  Tensor zero_point_out = tf_long.zeros({2, 1});
  Tensor expected_scale = tf_double.make({2, 1}, {0.00666667, 0.0113725485});
  Tensor expected_zero_point = tf_long.make({2, 1}, {-53, -58});

  choose_qparams_per_token_asymmetric_out(
      input, ScalarType::Float, scale_out, zero_point_out);

  EXPECT_TENSOR_CLOSE_WITH_TOL(scale_out, expected_scale, 1e-4, 1e-4);
  EXPECT_TENSOR_EQ(zero_point_out, expected_zero_point);
}

TEST(OpChooseQparamsPerTokenAsymmetricTensorOutTest, ExtraDimFloat) {
  et_pal_init();
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor input = tf_float.make({1, 2, 3}, {-0.5, 0.3, 1.2, 0.1, -0.8, 2.1});
  Tensor scale_out = tf_double.zeros({1, 2, 1});
  Tensor zero_point_out = tf_long.zeros({1, 2, 1});
  Tensor expected_scale = tf_double.make({1, 2, 1}, {0.00666667, 0.0113725485});
  Tensor expected_zero_point = tf_long.make({1, 2, 1}, {-53, -58});

  choose_qparams_per_token_asymmetric_out(
      input, ScalarType::Float, scale_out, zero_point_out);

  EXPECT_TENSOR_CLOSE_WITH_TOL(scale_out, expected_scale, 1e-4, 1e-4);
  EXPECT_TENSOR_EQ(zero_point_out, expected_zero_point);
}

TEST(OpChooseQparamsPerTokenAsymmetricTensorOutTest, LargeArray) {
  et_pal_init();
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor input = tf_float.make(
      {5, 17},
      {0.41654,  0.26599, 0.4141,   0.83809,  0.02938,  0.12199, 0.53667,
       0.799,    0.6606,  0.46657,  0.66142,  0.71787,  0.56098, 0.30202,
       0.059377, 0.85473, 0.8017,   0.2703,   0.44299,  0.49045, 0.75581,
       0.24429,  0.43906, 0.78652,  0.83885,  0.31034,  0.76534, 0.74422,
       0.62549,  0.80006, 0.38144,  0.70652,  0.33553,  0.89136, 0.49126,
       0.072916, 0.75654, 0.82057,  0.083848, 0.29753,  0.62718, 0.95579,
       0.83097,  0.47293, 0.15666,  0.6248,   0.21672,  0.14626, 0.71834,
       0.93664,  0.23382, 0.68931,  0.70866,  0.60545,  0.98648, 0.30335,
       0.62439,  0.19195, 0.1923,   0.75638,  0.81114,  0.34778, 0.0070671,
       0.50918,  0.19698, 0.19969,  0.57687,  0.062786, 0.18447, 0.22961,
       0.29656,  0.25486, 0.75965,  0.11328,  0.86468,  0.21264, 0.99591,
       0.75231,  0.97834, 0.042441, 0.39978,  0.9633,   0.9297,  0.12188,
       0.73564});
  Tensor scale_out = tf_double.zeros({5, 1});
  Tensor zero_point_out = tf_long.zeros({5, 1});
  Tensor expected_scale = tf_double.make(
      {5, 1}, {0.0033519, 0.0034955, 0.0037482, 0.0038685, 0.0039055});
  Tensor expected_zero_point =
      tf_long.make({5, 1}, {-128, -128, -128, -128, -128});

  choose_qparams_per_token_asymmetric_out(
      input, ScalarType::Float, scale_out, zero_point_out);

  EXPECT_TENSOR_CLOSE_WITH_TOL(scale_out, expected_scale, 1e-5, 1e-5);
  EXPECT_TENSOR_EQ(zero_point_out, expected_zero_point);
}

TEST(OpChooseQparamsPerTokenAsymmetricTensorOutTest, DynamicShapeFloat) {
  et_pal_init();
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Double> tf_double;
  TensorFactory<ScalarType::Long> tf_long;

  Tensor input = tf_float.make({1, 2, 3}, {-0.5, 0.3, 1.2, 0.1, -0.8, 2.1});
  Tensor scale_out = tf_double.zeros(
      {1, 5, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor zero_point_out = tf_long.zeros(
      {1, 5, 1}, torch::executor::TensorShapeDynamism::DYNAMIC_BOUND);
  Tensor expected_scale = tf_double.make({1, 2, 1}, {0.00666667, 0.0113725485});
  Tensor expected_zero_point = tf_long.make({1, 2, 1}, {-53, -58});

  choose_qparams_per_token_asymmetric_out(
      input, ScalarType::Float, scale_out, zero_point_out);

  EXPECT_TENSOR_CLOSE_WITH_TOL(scale_out, expected_scale, 1e-4, 1e-4);
  EXPECT_TENSOR_EQ(zero_point_out, expected_zero_point);

  Tensor new_input = tf_float.make(
      {1, 5, 8},
      {5.2254,   5.6041,   5.7653,   -1.0126,  -0.86126, -0.1606,  -0.99196,
       -1.067,   5.5913,   5.7713,   5.4901,   -0.43128, -1.1759,  -0.60466,
       -0.82913, -0.73623, 5.4588,   5.4066,   5.2644,   -0.89692, -0.16866,
       -0.63169, -0.42352, -0.48866, 5.594,    5.5223,   5.5277,   -0.17658,
       -0.30669, -1.1777,  -0.65389, -0.36422, 5.6375,   5.1857,   5.0743,
       -0.46654, -0.43817, -0.41506, -0.94515, -0.60247});
  Tensor new_expected_scale = tf_double.make(
      {1, 5, 1}, {0.026793, 0.027244, 0.024924, 0.026556, 0.025814});
  Tensor new_expected_zero_point =
      tf_long.make({1, 5, 1}, {-88, -85, -92, -84, -91});

  choose_qparams_per_token_asymmetric_out(
      new_input, ScalarType::Float, scale_out, zero_point_out);

  EXPECT_TENSOR_CLOSE_WITH_TOL(scale_out, new_expected_scale, 1e-4, 1e-4);
  EXPECT_TENSOR_EQ(zero_point_out, new_expected_zero_point);
}
