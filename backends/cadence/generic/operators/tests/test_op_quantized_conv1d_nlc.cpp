/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_quantized_conv1d_nlc.h>

#include <array>

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <gtest/gtest.h>
#include <optional>

namespace impl {
namespace generic {
namespace native {
namespace {

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::testing::TensorFactory;

class GenericQuantizedConv1dNlcPerChannelTest : public OperatorTest {
 protected:
  static IntArrayRef ref(const std::array<int64_t, 1>& a) {
    return IntArrayRef(a.data(), a.size());
  }
};

// NLC layout: input [N, L, C], weight [OC, WW, WC].
// Same numeric construction as NCL, reinterpreted for NLC.
//   input [1, 4, 1]  = [1, 2, 3, 4] (single input channel, length 4)
//   weight [3, 1, 1] = [2, 2, 2]     (oc=3, ww=1, wc=1)
//   bias  [3]        = [0, 0, 0]
//   weight_zero_point [3] = [1, 0, 2]
//     -> effective weight per output channel: 1, 2, 0
//   bias_scale [3]        = [1.0f, 5.0f, 10.0f]
//
// Output NLC [1, 4, 3], laid out as position-major:
//   pos 0: [ 1, 10, 0]
//   pos 1: [ 2, 20, 0]
//   pos 2: [ 3, 30, 0]
//   pos 3: [ 4, 40, 0]
TEST_F(GenericQuantizedConv1dNlcPerChannelTest, PerChannelCorrectnessInt8) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Int> tf_int32;
  TensorFactory<ScalarType::Float> tf_float;

  Tensor input = tf_int8.make({1, 4, 1}, {1, 2, 3, 4});
  Tensor weight = tf_int8.make({3, 1, 1}, {2, 2, 2});
  Tensor bias = tf_int32.zeros({3});
  Tensor weight_zero_point = tf_int32.make({3}, {1, 0, 2});
  Tensor bias_scale = tf_float.make({3}, {1.0f, 5.0f, 10.0f});
  Tensor out_multiplier = tf_int32.make({1}, {0});
  Tensor out_shift = tf_int32.make({1}, {0});
  Tensor out = tf_int8.zeros({1, 4, 3});

  std::array<int64_t, 1> stride_arr = {1};
  std::array<int64_t, 1> padding_arr = {0};
  std::array<int64_t, 1> dilation_arr = {1};

  quantized_conv1d_nlc_out(
      context_,
      input,
      weight,
      bias,
      ref(stride_arr),
      ref(padding_arr),
      ref(dilation_arr),
      /*groups=*/1,
      /*input_zero_point=*/0,
      weight_zero_point,
      bias_scale,
      /*output_scale=*/1.0,
      /*output_zero_point=*/0,
      out_multiplier,
      out_shift,
      out);

  Tensor expected = tf_int8.make(
      {1, 4, 3},
      {
          /*pos 0*/ 1,
          10,
          0,
          /*pos 1*/ 2,
          20,
          0,
          /*pos 2*/ 3,
          30,
          0,
          /*pos 3*/ 4,
          40,
          0,
      });
  EXPECT_TENSOR_EQ(out, expected);
}

// Degeneracy: length-1 qparam tensors on .out must match the
// existing _out kernel.
TEST_F(GenericQuantizedConv1dNlcPerChannelTest, LengthOneMatchesQuantizedOut) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Int> tf_int32;
  TensorFactory<ScalarType::Float> tf_float;

  Tensor input = tf_int8.make({1, 4, 1}, {1, 2, 3, 4});
  Tensor weight = tf_int8.make({3, 1, 1}, {2, 2, 2});
  Tensor bias = tf_int32.zeros({3});

  Tensor weight_zero_point = tf_int32.make({1}, {1});
  Tensor bias_scale = tf_float.make({1}, {5.0f});
  Tensor out_multiplier = tf_int32.make({1}, {0});
  Tensor out_shift = tf_int32.make({1}, {0});

  std::array<int64_t, 1> stride_arr = {1};
  std::array<int64_t, 1> padding_arr = {0};
  std::array<int64_t, 1> dilation_arr = {1};

  Tensor out_pc = tf_int8.zeros({1, 4, 3});
  quantized_conv1d_nlc_out(
      context_,
      input,
      weight,
      bias,
      ref(stride_arr),
      ref(padding_arr),
      ref(dilation_arr),
      /*groups=*/1,
      /*input_zero_point=*/0,
      weight_zero_point,
      bias_scale,
      /*output_scale=*/1.0,
      /*output_zero_point=*/0,
      out_multiplier,
      out_shift,
      out_pc);

  // The scalar-qparam overload is the reference: a length-1 qparam vector has
  // to mean exactly what a scalar qparam means.
  Tensor out_ref = tf_int8.zeros({1, 4, 3});
  quantized_conv1d_nlc_per_tensor_out(
      context_,
      input,
      weight,
      bias,
      ref(stride_arr),
      ref(padding_arr),
      ref(dilation_arr),
      /*groups=*/1,
      /*input_zero_point=*/0,
      /*weight_zero_point=*/1,
      /*bias_scale=*/5.0,
      /*output_scale=*/1.0,
      /*output_zero_point=*/0,
      /*out_multiplier=*/0,
      /*out_shift=*/0,
      /*offset=*/std::nullopt,
      out_ref);

  EXPECT_TENSOR_EQ(out_pc, out_ref);
}

} // namespace
} // namespace native
} // namespace generic
} // namespace impl
