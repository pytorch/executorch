/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_quantized_linear.h>

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <gtest/gtest.h>

namespace impl {
namespace generic {
namespace native {
namespace {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::testing::TensorFactory;

class GenericQuantizedLinearPerChannelTest : public OperatorTest {};

// Per-channel correctness. Uses distinct out_multiplier/out_shift per output
// channel so a regression that reads `[0]` for every channel produces visibly
// wrong values on channels 1..N-1.
//
// Setup (all zero_points = 0):
//   src (int8, [1,2])       = [[10, 10]]
//   weight (int8, [3,2])    = [[1, 1], [2, 2], [4, 4]]  -> distinct rows
//   bias (int32, [3])       = [0, 0, 0]
//   weight_zero_point ([1]) = [0]        (per-tensor weight zp)
//   out_multiplier ([3])    = per-channel, produces requant scales
//                             [1/16, 1/8, 1/4]
//   out_shift ([3])         = [0, 0, 0]
//
// Sums (integer accum from formula bias + sum_k (src-zp)*(w-wzp)):
//   ch 0: 10*1 + 10*1 = 20
//   ch 1: 10*2 + 10*2 = 40
//   ch 2: 10*4 + 10*4 = 80
//
// requant_scale[j] = out_multiplier[j] / 2^31 * 2^out_shift[j]
//   ch 0: 2^27/2^31 * 1     = 1/16 = 0.0625
//   ch 1: 1/8               = 0.125
//   ch 2: 1/4               = 0.25
//
// out[j] = clamp(out_zp + nearbyint(requant_scale[j] * sum))
//   ch 0: 20*0.0625 = 1.25   -> 1
//   ch 1: 40*0.125  = 5.0    -> 5
//   ch 2: 80*0.25   = 20.0   -> 20
//
// A buggy implementation that always reads out_multiplier/out_shift[0] would
// use scale=0.0625 for every channel, yielding [1, 2, 5] instead of [1, 5, 20].
TEST_F(GenericQuantizedLinearPerChannelTest, PerChannelCorrectnessInt8) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Int> tf_int32;

  Tensor src = tf_int8.make({1, 2}, {10, 10});
  Tensor weight = tf_int8.make({3, 2}, {1, 1, 2, 2, 4, 4});
  Tensor bias = tf_int32.zeros({3});
  Tensor weight_zero_point = tf_int32.make({1}, {0});
  // Positive Q31 multipliers by the existing kernel convention:
  // requant_scale = out_multiplier / 2^31 (the two negatives in the C++
  // formula cancel). See existing tests, e.g. HiFi conv2d_out.
  Tensor out_multiplier = tf_int32.make(
      {3},
      {
          (1 << 27), // 1/16
          (1 << 28), // 1/8
          (1 << 29), // 1/4
      });
  Tensor out_shift = tf_int32.make({3}, {0, 0, 0});
  Tensor out = tf_int8.zeros({1, 3});

  quantized_linear_out(
      context_,
      src,
      weight,
      bias,
      /*src_zero_point=*/0,
      weight_zero_point,
      out_multiplier,
      out_shift,
      /*out_zero_point=*/0,
      /*offset=*/std::nullopt,
      out);

  Tensor expected = tf_int8.make({1, 3}, {1, 5, 20});
  EXPECT_TENSOR_EQ(out, expected);
}

// Degeneracy: length-1 qparam tensors on the .out entry point must
// match the existing per_tensor_out kernel with equivalent scalar qparams.
// Pins the stride-selection logic (numel() > 1 ? 1 : 0).
TEST_F(GenericQuantizedLinearPerChannelTest, LengthOneMatchesPerTensor) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Int> tf_int32;

  Tensor src = tf_int8.make({1, 2}, {10, 10});
  Tensor weight = tf_int8.make({3, 2}, {1, 1, 2, 2, 4, 4});
  Tensor bias = tf_int32.zeros({3});
  Tensor weight_zero_point = tf_int32.make({1}, {0});
  // Single scalar-equivalent qparam (positive Q31 multiplier).
  const int32_t scalar_multiplier = (1 << 28); // 1/8
  const int32_t scalar_shift = 0;
  Tensor out_multiplier = tf_int32.make({1}, {scalar_multiplier});
  Tensor out_shift = tf_int32.make({1}, {scalar_shift});

  Tensor out_pc = tf_int8.zeros({1, 3});
  quantized_linear_out(
      context_,
      src,
      weight,
      bias,
      /*src_zero_point=*/0,
      weight_zero_point,
      out_multiplier,
      out_shift,
      /*out_zero_point=*/0,
      /*offset=*/std::nullopt,
      out_pc);

  Tensor out_pt = tf_int8.zeros({1, 3});
  quantized_linear_per_tensor_out(
      context_,
      src,
      weight,
      bias,
      /*src_zero_point=*/0,
      /*weight_zero_point=*/0,
      /*out_multiplier=*/scalar_multiplier,
      /*out_shift=*/scalar_shift,
      /*out_zero_point=*/0,
      /*offset=*/std::nullopt,
      out_pt);

  EXPECT_TENSOR_EQ(out_pc, out_pt);
}

// The uint8 typed entry. The kernel computes -out_multiplier / (1 << 31), and
// 1 << 31 is INT_MIN, so the scale is +out_multiplier / 2^31, as
// quantize_tensor_multiplier produces. aot/ref_implementations.py computes
// -out_multiplier / 2^31, so the expected values come from it with the
// multiplier negated: (src - 128) @ (weight - 128)^T + bias =
// [[-3, 80], [-4, 0]], times 1/4, plus 100.
TEST_F(GenericQuantizedLinearPerChannelTest, Asym8uPerTensor) {
  TensorFactory<ScalarType::Byte> tf_uint8;
  TensorFactory<ScalarType::Int> tf_int32;
  Tensor src = tf_uint8.make({2, 3}, {130, 125, 140, 120, 128, 128});
  Tensor weight = tf_uint8.make({2, 3}, {129, 131, 128, 127, 126, 135});
  Tensor bias = tf_int32.make({2}, {4, -8});
  Tensor out = tf_uint8.zeros({2, 2});

  quantized_linear_asym8uxasym8u_asym8u_per_tensor_out(
      context_,
      src,
      weight,
      bias,
      /*src_zero_point=*/128,
      /*weight_zero_point=*/128,
      /*out_multiplier=*/(1 << 29),
      /*out_shift=*/0,
      /*out_zero_point=*/100,
      /*offset=*/std::nullopt,
      out);

  EXPECT_TENSOR_EQ(out, tf_uint8.make({2, 2}, {99, 120, 99, 100}));
}

} // namespace
} // namespace native
} // namespace generic
} // namespace impl
