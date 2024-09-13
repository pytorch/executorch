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

#include <gtest/gtest.h>

using namespace ::testing;
using exec_aten::optional;
using exec_aten::ScalarType;
using exec_aten::Tensor;
using torch::executor::testing::TensorFactory;

::std::tuple<Tensor&, Tensor&, Tensor&> op_native_group_norm_out(
    const Tensor& input,
    const optional<Tensor>& weight,
    const optional<Tensor>& bias,
    int64_t N,
    int64_t C,
    int64_t HxW,
    int64_t group,
    double eps,
    Tensor& out0,
    Tensor& out1,
    Tensor& out2) {
  executorch::runtime::KernelRuntimeContext context{};
  return torch::executor::aten::native_group_norm_outf(
      context, input, weight, bias, N, C, HxW, group, eps, out0, out1, out2);
}

TEST(OpNativeGroupNormOutTest, SmokeTest) {
  TensorFactory<ScalarType::Float> tfFloat;

  Tensor input = tfFloat.make(
      {5, 6, 2, 2},
      {-0.8125, 0.0625,  -2.7500, -3.0625, -1.1250, -2.1250, -1.3125, -4.0625,
       2.8125,  -2.0625, 4.2500,  3.5000,  -0.3750, 1.6250,  4.3125,  -1.0625,
       -2.8750, 3.3750,  4.9375,  4.0625,  -3.0625, -1.8750, -2.7500, -2.5625,
       -0.1875, -3.0000, -2.7500, 0.6875,  -3.2500, -3.1875, 1.0000,  -4.6250,
       -0.1875, -1.7500, 4.5000,  -1.8750, -2.6875, 4.8125,  -3.8125, -2.9375,
       -1.1875, 2.8750,  0.7500,  2.8750,  1.1250,  -0.6250, -2.2500, -3.7500,
       3.2500,  -0.3750, -2.0625, -4.7500, 2.0625,  3.0000,  -3.1875, -4.1250,
       -3.7500, 1.2500,  -2.3125, 1.5625,  3.1250,  0.3125,  3.2500,  -2.7500,
       -3.8125, -4.2500, -4.3125, -0.5625, -0.4375, 2.9375,  -1.3750, -0.6250,
       -2.5625, -4.5625, 0.1250,  -3.5000, -5.0000, -1.0000, -4.6875, -0.6875,
       1.1250,  1.8750,  -4.5000, 4.3125,  4.5625,  0.2500,  -3.6250, 4.5625,
       -3.5000, -2.1250, -3.6250, -2.9375, 3.6875,  3.9375,  4.3750,  3.0625,
       2.4375,  2.0625,  -2.4375, -3.9375, 3.6875,  2.7500,  -0.8750, -0.9375,
       2.7500,  -2.4375, -2.3750, -0.9375, -4.8750, 0.1875,  3.5000,  -2.0000,
       -0.2500, -2.7500, 0.3125,  1.2500,  -0.5625, 0.0000,  1.8125,  1.0625});
  optional<Tensor> weight =
      tfFloat.make({6}, {4.5625, -2.8750, -0.6875, 0.5625, -2.0625, -2.7500});
  optional<Tensor> bias =
      tfFloat.make({6}, {-0.5000, -2.7500, 1.1875, 3.6875, 3.8125, 4.6875});
  double eps = 1e-5;
  Tensor out0 = tfFloat.zeros({5, 6, 2, 2});
  Tensor out1 = tfFloat.zeros({5, 3});
  Tensor out2 = tfFloat.zeros({5, 3});
  Tensor out0_expected = tfFloat.make(
      {5, 6, 2, 2},
      {3.419882,  6.578348,  -3.573864, -4.701888, -4.509254, -2.234663,
       -4.082768, 2.172355,  0.838826,  2.270225,  0.416747,  0.636962,
       3.207030,  3.687500,  4.333131,  3.041869,  5.547079,  1.649148,
       0.674665,  1.220376,  7.156189,  6.168714,  6.896327,  6.740410,
       3.509863,  -3.022041, -2.441427, 5.542011,  -0.794903, -0.886369,
       -7.014627, 1.217361,  1.120617,  1.463606,  0.091652,  1.491045,
       3.293219,  4.640229,  3.091168,  3.248319,  4.895990,  1.114683,
       3.092597,  1.114683,  3.262238,  5.434066,  7.450763,  9.312329,
       5.570122,  0.101119,  -2.444796, -6.499403, -5.446074, -6.337338,
       -0.454995, 0.436269,  2.228491,  0.871598,  1.838385,  0.786793,
       4.362284,  3.737805,  4.390039,  3.057817,  5.814659,  6.202621,
       6.258044,  2.932658,  3.366583,  -0.623879, 4.475045,  3.588276,
       -0.082914, -4.936279, 6.438795,  -2.357929, 0.714463,  -5.402106,
       0.236606,  -5.879963, 1.176247,  1.021916,  2.333727,  0.520341,
       4.275447,  3.549392,  2.896994,  4.275447,  6.120910,  5.298480,
       6.195676,  5.784461,  2.033296,  1.833920,  1.485010,  2.531738,
       3.193988,  2.532378,  -5.406940, -8.053379, -6.467402, -5.425139,
       -1.395059, -1.325575, 0.266062,  1.622680,  1.606336,  1.230405,
       2.809896,  3.893110,  4.601880,  3.425055,  4.374411,  8.283354,
       3.494898,  2.029045,  6.088204,  4.915522,  1.136877,  2.700454});
  Tensor out1_expected = tfFloat.make(
      {5, 3},
      {-1.89843750,
       1.62500000,
       -0.09375000,
       -1.91406250,
       -0.49218744,
       -0.02343750,
       -0.77343756,
       0.08593753,
       -1.55468738,
       -2.73437500,
       1.07031238,
       0.35937503,
       0.34374997,
       -0.77343750,
       0.10937499});
  Tensor out2_expected = tfFloat.make(
      {5, 3},
      {0.79116172,
       0.42708409,
       0.30238494,
       0.50903118,
       0.31929117,
       0.45128885,
       0.33067191,
       0.39473253,
       0.42994878,
       0.53187561,
       0.29930803,
       0.29000264,
       0.38669431,
       0.38038814,
       0.75809801});
  op_native_group_norm_out(
      input, weight, bias, 5, 6, 4, 3, eps, out0, out1, out2);
  EXPECT_TENSOR_CLOSE(out0, out0_expected);
  EXPECT_TENSOR_CLOSE(out1, out1_expected);
  EXPECT_TENSOR_CLOSE(out2, out2_expected);
}
