/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/xnnpack/runtime/utils/utils.h>
#include <executorch/extension/aten_util/aten_bridge.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/platform/runtime.h>

using namespace ::testing;

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::Error;
using executorch::runtime::testing::TensorFactory;
namespace utils = executorch::backends::xnnpack::utils;

TEST(TestUtils, choose_quantization_params) {
  Error e;
  utils::QuantizationParams qparams;
  float min = -128.0 * 10.0;
  float max = +127.0 * 10.0;
  e = utils::ChooseQuantizationParams(
      min, max, 0, 255, qparams, false, false, false);
  ASSERT_EQ(e, Error::Ok);
  ASSERT_EQ(qparams.zero_point, 128);
  ASSERT_EQ(qparams.scale, 10.0);
}

TEST(TestUtils, choose_quantization_params_fails) {
  executorch::runtime::runtime_init();
  Error e;
  utils::QuantizationParams qparams;
  float min = -128.0 * 10.0;
  float max = +127.0 * 10.0;
  e = utils::ChooseQuantizationParams(
      max, min, 0, 255, qparams, false, false, false);
  ASSERT_EQ(e, Error::Internal);
}

TEST(TestUtils, quantize_per_tensor) {
  TensorFactory<ScalarType::Float> tf;
  const Tensor input = tf.full({3, 5}, 4);
  double scale = 0.5;
  int zero_point = 127;
  TensorFactory<ScalarType::QUInt8> tfo;
  Tensor output = tfo.zeros({3, 5});
  // 4 / 0.5 + 127
  auto at_tensor = at::full({3, 5}, 4.f);
  auto at_expected = at::quantize_per_tensor(
      at_tensor, scale, zero_point, at::ScalarType::QUInt8);
  Tensor expected = tfo.zeros_like(output);
  at_expected = at_expected.contiguous();
  executorch::extension::alias_etensor_to_attensor(at_expected, expected);
  Error e = utils::QuantizePerTensor(input, output, scale, zero_point);
  ASSERT_EQ(e, Error::Ok);
  EXPECT_TENSOR_EQ(output, expected);
}

TEST(TestUtils, generate_requantizeation_scale) {
  TensorFactory<ScalarType::Float> tf;
  const Tensor weight_scales = tf.full({3, 5}, 4.0);
  float input_scale = 2.0;
  float output_scale = 3.0;
  std::vector<float> req_scales(15, 0);
  Error e = utils::GenerateRequantizationScale(
      weight_scales, input_scale, output_scale, req_scales);
  ASSERT_EQ(e, Error::Ok);
  for (auto m : req_scales) {
    EXPECT_FLOAT_EQ(m, 4.0 * 2.0 / 3.0);
  }
}

TEST(TestUtils, get_min_max) {
  TensorFactory<ScalarType::Float> tf;
  float min, max;

  float val = 4.12345;
  const Tensor ft = tf.full({3, 5}, val);
  std::tie(min, max) = utils::GetMinMax(ft);
  EXPECT_FLOAT_EQ(min, val);
  EXPECT_FLOAT_EQ(max, val);

  const Tensor ft_min = tf.make(
      {2, 1},
      {std::numeric_limits<float>::min(), std::numeric_limits<float>::max()});
  std::tie(min, max) = utils::GetMinMax(ft_min);
  EXPECT_FLOAT_EQ(min, std::numeric_limits<float>::min());
  EXPECT_FLOAT_EQ(max, std::numeric_limits<float>::max());

  const Tensor ft_lowest = tf.make(
      {2, 1},
      {std::numeric_limits<float>::lowest(),
       std::numeric_limits<float>::max()});
  std::tie(min, max) = utils::GetMinMax(ft_lowest);
  EXPECT_FLOAT_EQ(min, std::numeric_limits<float>::lowest());
  EXPECT_FLOAT_EQ(max, std::numeric_limits<float>::max());

  const Tensor ft_random = tf.make({5, 1}, {-2.2, -1.1, 0, 1.1, 2.2});
  std::tie(min, max) = utils::GetMinMax(ft_random);
  EXPECT_FLOAT_EQ(min, -2.2);
  EXPECT_FLOAT_EQ(max, 2.2);
}
