/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_quantized_fully_connected.h>

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

class GenericQuantizedFullyConnectedPerChannelTest : public OperatorTest {};

// Per-channel correctness. Same construction as the linear test, since FC
// dispatches to the same underlying `quantized_linear_` kernel. See that test
// for the derivation of the expected values.
TEST_F(
    GenericQuantizedFullyConnectedPerChannelTest,
    PerChannelCorrectnessInt8) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Int> tf_int32;

  Tensor in = tf_int8.make({1, 2}, {10, 10});
  Tensor weight = tf_int8.make({3, 2}, {1, 1, 2, 2, 4, 4});
  Tensor bias = tf_int32.zeros({3});
  Tensor weight_zero_point = tf_int32.make({1}, {0});
  Tensor out_multiplier = tf_int32.make(
      {3},
      {
          (1 << 27), // 1/16
          (1 << 28), // 1/8
          (1 << 29), // 1/4
      });
  Tensor out_shift = tf_int32.make({3}, {0, 0, 0});
  Tensor out = tf_int8.zeros({1, 3});

  quantized_fully_connected_out(
      context_,
      in,
      weight,
      bias,
      /*in_zero_point=*/0,
      weight_zero_point,
      out_multiplier,
      out_shift,
      /*out_zero_point=*/0,
      /*offset=*/std::nullopt,
      out);

  // ch 0: 20 * 0.0625 = 1.25 -> 1
  // ch 1: 40 * 0.125  = 5.0  -> 5
  // ch 2: 80 * 0.25   = 20.0 -> 20
  Tensor expected = tf_int8.make({1, 3}, {1, 5, 20});
  EXPECT_TENSOR_EQ(out, expected);
}

// Degeneracy: length-1 qparam tensors on the .out entry point must
// match the existing per_tensor_out kernel with equivalent scalar qparams.
TEST_F(
    GenericQuantizedFullyConnectedPerChannelTest,
    LengthOneMatchesPerTensor) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Int> tf_int32;

  Tensor in = tf_int8.make({1, 2}, {10, 10});
  Tensor weight = tf_int8.make({3, 2}, {1, 1, 2, 2, 4, 4});
  Tensor bias = tf_int32.zeros({3});
  Tensor weight_zero_point = tf_int32.make({1}, {0});
  const int32_t scalar_multiplier = (1 << 28); // 1/8
  const int32_t scalar_shift = 0;
  Tensor out_multiplier = tf_int32.make({1}, {scalar_multiplier});
  Tensor out_shift = tf_int32.make({1}, {scalar_shift});

  Tensor out_pc = tf_int8.zeros({1, 3});
  quantized_fully_connected_out(
      context_,
      in,
      weight,
      bias,
      /*in_zero_point=*/0,
      weight_zero_point,
      out_multiplier,
      out_shift,
      /*out_zero_point=*/0,
      /*offset=*/std::nullopt,
      out_pc);

  Tensor out_pt = tf_int8.zeros({1, 3});
  quantized_fully_connected_per_tensor_out(
      context_,
      in,
      weight,
      bias,
      /*in_zero_point=*/0,
      /*weight_zero_point=*/0,
      /*out_multiplier=*/scalar_multiplier,
      /*out_shift=*/scalar_shift,
      /*out_zero_point=*/0,
      /*offset=*/std::nullopt,
      out_pt);

  EXPECT_TENSOR_EQ(out_pc, out_pt);
}

// The uint8 typed entry, on the first row of the linear test's input.
// Expected values are aot/ref_implementations.py's with the multiplier negated,
// for the reason test_op_quantized_linear.cpp gives.
TEST_F(GenericQuantizedFullyConnectedPerChannelTest, Asym8uPerTensor) {
  TensorFactory<ScalarType::Byte> tf_uint8;
  TensorFactory<ScalarType::Int> tf_int32;
  Tensor in = tf_uint8.make({1, 3}, {130, 125, 140});
  Tensor weight = tf_uint8.make({2, 3}, {129, 131, 128, 127, 126, 135});
  Tensor bias = tf_int32.make({2}, {4, -8});
  Tensor out = tf_uint8.zeros({1, 2});

  quantized_fully_connected_asym8uxasym8u_asym8u_per_tensor_out(
      context_,
      in,
      weight,
      bias,
      /*in_zero_point=*/128,
      /*weight_zero_point=*/128,
      /*out_multiplier=*/(1 << 29),
      /*out_shift=*/0,
      /*out_zero_point=*/100,
      /*offset=*/std::nullopt,
      out);

  EXPECT_TENSOR_EQ(out, tf_uint8.make({1, 2}, {99, 120}));
}

} // namespace
} // namespace native
} // namespace generic
} // namespace impl
