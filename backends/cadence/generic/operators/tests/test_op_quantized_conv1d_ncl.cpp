/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_quantized_conv1d_ncl.h>

#include <array>

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <gtest/gtest.h>

namespace impl {
namespace generic {
namespace native {
namespace {

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::testing::TensorFactory;

class GenericQuantizedConv1dNclPerChannelTest : public OperatorTest {
 protected:
  static IntArrayRef ref(const std::array<int64_t, 1>& a) {
    return IntArrayRef(a.data(), a.size());
  }
};

// NCL layout: input [N, C, L].
// Setup (int8, all zero_points = 0, output_scale = 1.0):
//   input  [1, 1, 4] = [[[1, 2, 3, 4]]]
//   weight [3, 1, 1] = [[[2]], [[2]], [[2]]]           (kernel_size = 1)
//   bias   [3]       = [0, 0, 0]
//   weight_zero_point [3] = [1, 0, 2]
//     -> effective weight per output channel: 1, 2, 0
//   bias_scale [3]        = [1.0f, 5.0f, 10.0f]
//
// Per-position accumulator per output channel:
//   ch 0: acc = input * 1  -> [1, 2, 3, 4]
//   ch 1: acc = input * 2  -> [2, 4, 6, 8]
//   ch 2: acc = input * 0  -> [0, 0, 0, 0]
//
// Output NCL [1, 3, 4]:
//   ch 0 (bs=1.0):  [ 1,  2,  3,  4]
//   ch 1 (bs=5.0):  [10, 20, 30, 40]
//   ch 2 (bs=10.0): [ 0,  0,  0,  0]
TEST_F(GenericQuantizedConv1dNclPerChannelTest, PerChannelCorrectnessInt8) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Int> tf_int32;
  TensorFactory<ScalarType::Float> tf_float;

  Tensor input = tf_int8.make({1, 1, 4}, {1, 2, 3, 4});
  Tensor weight = tf_int8.make({3, 1, 1}, {2, 2, 2});
  Tensor bias = tf_int32.zeros({3});
  Tensor weight_zero_point = tf_int32.make({3}, {1, 0, 2});
  Tensor bias_scale = tf_float.make({3}, {1.0f, 5.0f, 10.0f});
  Tensor out_multiplier = tf_int32.make({1}, {0});
  Tensor out_shift = tf_int32.make({1}, {0});
  Tensor out = tf_int8.zeros({1, 3, 4});

  std::array<int64_t, 1> stride_arr = {1};
  std::array<int64_t, 1> padding_arr = {0};
  std::array<int64_t, 1> dilation_arr = {1};

  quantized_conv1d_ncl_out(
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
      {1, 3, 4},
      {
          /*ch 0*/ 1,
          2,
          3,
          4,
          /*ch 1*/ 10,
          20,
          30,
          40,
          /*ch 2*/ 0,
          0,
          0,
          0,
      });
  EXPECT_TENSOR_EQ(out, expected);
}

// Degeneracy: length-1 qparam tensors on .out must match the
// existing _out kernel (which itself only uses element 0).
TEST_F(GenericQuantizedConv1dNclPerChannelTest, LengthOneMatchesQuantizedOut) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Int> tf_int32;
  TensorFactory<ScalarType::Float> tf_float;

  Tensor input = tf_int8.make({1, 1, 4}, {1, 2, 3, 4});
  Tensor weight = tf_int8.make({3, 1, 1}, {2, 2, 2});
  Tensor bias = tf_int32.zeros({3});

  Tensor weight_zero_point = tf_int32.make({1}, {1});
  Tensor bias_scale = tf_float.make({1}, {5.0f});
  Tensor out_multiplier = tf_int32.make({1}, {0});
  Tensor out_shift = tf_int32.make({1}, {0});

  std::array<int64_t, 1> stride_arr = {1};
  std::array<int64_t, 1> padding_arr = {0};
  std::array<int64_t, 1> dilation_arr = {1};

  Tensor out_pc = tf_int8.zeros({1, 3, 4});
  quantized_conv1d_ncl_out(
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
  Tensor out_ref = tf_int8.zeros({1, 3, 4});
  quantized_conv1d_ncl_per_tensor_out(
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
      out_ref);

  EXPECT_TENSOR_EQ(out_pc, out_ref);
}

} // namespace
} // namespace native
} // namespace generic
} // namespace impl
