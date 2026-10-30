/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>

#include <executorch/backends/cadence/hifi/operators/operators.h>

namespace impl {
namespace HiFi {
namespace native {
namespace {

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::testing::TensorFactory;

// Scales are powers of two and inputs are symmetric around their mean, so
// every row normalizes to exactly +/-1 (eps is negligible).
constexpr double kEps = 1e-9;

class HiFiQuantizedLayerNormTest : public OperatorTest {};

TEST_F(HiFiQuantizedLayerNormTest, Int16LargeMagnitudePerTensor) {
  TensorFactory<ScalarType::Short> tf_short;
  TensorFactory<ScalarType::Float> tf_float;
  const std::vector<int64_t> normalized_shape{8};

  // Per-row sum of squares is 8 * 30000^2, well past int32 range.
  const Tensor input = tf_short.make(
      {2, 8},
      {30000,
       -30000,
       30000,
       -30000,
       30000,
       -30000,
       30000,
       -30000,
       -30000,
       30000,
       -30000,
       30000,
       -30000,
       30000,
       -30000,
       30000});
  const Tensor weight = tf_float.ones({8});
  const Tensor bias = tf_float.zeros({8});
  Tensor out = tf_short.zeros({2, 8});

  quantized_layer_norm_per_tensor_out(
      context_,
      input,
      /*in_scale=*/1.0 / 1024,
      /*in_zero_point=*/0,
      IntArrayRef(normalized_shape.data(), normalized_shape.size()),
      weight,
      bias,
      kEps,
      /*output_scale=*/1.0 / 16384,
      /*output_zero_point=*/0,
      out);

  const Tensor expected = tf_short.make(
      {2, 8},
      {16384,
       -16384,
       16384,
       -16384,
       16384,
       -16384,
       16384,
       -16384,
       -16384,
       16384,
       -16384,
       16384,
       -16384,
       16384,
       -16384,
       16384});
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(HiFiQuantizedLayerNormTest, Int16LargeMeanSmallSpreadWithZeroPoints) {
  TensorFactory<ScalarType::Short> tf_short;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Long> tf_long;
  const std::vector<int64_t> normalized_shape{8};

  // Centered values are 21000 +/- 4: the mean dwarfs the spread, which breaks
  // an fp32 E[x^2] - E[x]^2 variance.
  const Tensor input = tf_short.make(
      {1, 8}, {20004, 19996, 20004, 19996, 20004, 19996, 20004, 19996});
  const Tensor in_scale = tf_float.make({1}, {1.0f / 64});
  const Tensor in_zero_point = tf_long.make({1}, {-1000});
  const Tensor weight = tf_float.full({8}, 2.0f);
  const Tensor bias = tf_float.full({8}, 0.5f);
  Tensor out = tf_short.zeros({1, 8});

  quantized_layer_norm_out(
      context_,
      input,
      in_scale,
      in_zero_point,
      IntArrayRef(normalized_shape.data(), normalized_shape.size()),
      weight,
      bias,
      kEps,
      /*output_scale=*/1.0 / 4096,
      /*output_zero_point=*/100,
      out);

  // (+1 * 2 + 0.5) * 4096 + 100 and (-1 * 2 + 0.5) * 4096 + 100.
  const Tensor expected = tf_short.make(
      {1, 8}, {10340, -6044, 10340, -6044, 10340, -6044, 10340, -6044});
  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(HiFiQuantizedLayerNormTest, Int8WithZeroPoint) {
  TensorFactory<ScalarType::Char> tf_char;
  TensorFactory<ScalarType::Float> tf_float;
  const std::vector<int64_t> normalized_shape{4};

  const Tensor input = tf_char.make({1, 4}, {10, 30, 10, 30});
  const Tensor weight = tf_float.ones({4});
  const Tensor bias = tf_float.zeros({4});
  Tensor out = tf_char.zeros({1, 4});

  quantized_layer_norm_per_tensor_out(
      context_,
      input,
      /*in_scale=*/1.0 / 8,
      /*in_zero_point=*/20,
      IntArrayRef(normalized_shape.data(), normalized_shape.size()),
      weight,
      bias,
      kEps,
      /*output_scale=*/1.0 / 64,
      /*output_zero_point=*/0,
      out);

  const Tensor expected = tf_char.make({1, 4}, {-64, 64, -64, 64});
  EXPECT_TENSOR_EQ(out, expected);
}

} // namespace
} // namespace native
} // namespace HiFi
} // namespace impl
