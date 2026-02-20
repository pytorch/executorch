/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/operators/operators.h>

#include <array>

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>

namespace impl {
namespace HiFi {
namespace native {
namespace {

using ::executorch::aten::Scalar;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::aten::TensorImpl;
using ::executorch::runtime::Error;
using ::executorch::runtime::KernelRuntimeContext;
using ::executorch::runtime::runtime_init;
using ::executorch::runtime::testing::TensorFactory;

class HiFiQuantizedConv2dTest : public OperatorTest {
 public:
 protected:
  void quantized_conv2d_nchw_out(
      const Tensor& input,
      const Tensor& weight,
      const Tensor& bias,
      ::executorch::aten::IntArrayRef stride,
      ::executorch::aten::IntArrayRef padding,
      ::executorch::aten::IntArrayRef dilation,
      int64_t groups,
      int64_t in_zero_point,
      const Tensor& weight_zero_point,
      const Tensor& bias_scale,
      double output_scale,
      int64_t output_zero_point,
      const Tensor& out_multiplier,
      const Tensor& out_shift,
      Tensor& output) {
    return ::impl::HiFi::native::quantized_conv2d_nchw_out(
        context_,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        in_zero_point,
        weight_zero_point,
        bias_scale,
        output_scale,
        output_zero_point,
        out_multiplier,
        out_shift,
        output);
  }

  void quantized_conv2d_nhwc_out(
      const Tensor& input,
      const Tensor& weight,
      const Tensor& bias,
      ::executorch::aten::IntArrayRef stride,
      ::executorch::aten::IntArrayRef padding,
      ::executorch::aten::IntArrayRef dilation,
      int64_t groups,
      int64_t in_zero_point,
      const Tensor& weight_zero_point,
      const Tensor& bias_scale,
      double output_scale,
      int64_t output_zero_point,
      const Tensor& out_multiplier,
      const Tensor& out_shift,
      Tensor& output) {
    return ::impl::HiFi::native::quantized_conv2d_nhwc_out(
        context_,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        in_zero_point,
        weight_zero_point,
        bias_scale,
        output_scale,
        output_zero_point,
        out_multiplier,
        out_shift,
        output);
  }
};

// Test quantized_conv2d_nchw_out with int16 activations and int8 weights
TEST_F(HiFiQuantizedConv2dTest, QuantizedConv2dNchwInt16Test) {
  TensorFactory<ScalarType::Short> tf_int16;
  TensorFactory<ScalarType::Int> tf_int32;
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;

  // Minimal test case: input [1, 2, 3, 3], kernel [1, 2, 2, 2] -> output [1, 1,
  // 2, 2] Small enough to verify by hand calculation
  //
  // Input Channel 0 (3x3):     Input Channel 1 (3x3):
  // 1  2  3                    1  1  1
  // 4  5  6                    1  1  1
  // 7  8  9                    1  1  1
  //
  // Weight Out Ch 0, In Ch 0:  Weight Out Ch 0, In Ch 1:
  // 1  0                       1  1
  // 0  1                       1  1
  //
  // Hand calculation for each output position:
  // (0,0): Ch0: 1*1+2*0+4*0+5*1=6,  Ch1: 1*1+1*1+1*1+1*1=4  -> 10
  // (0,1): Ch0: 2*1+3*0+5*0+6*1=8,  Ch1: 1*1+1*1+1*1+1*1=4  -> 12
  // (1,0): Ch0: 4*1+5*0+7*0+8*1=12, Ch1: 1*1+1*1+1*1+1*1=4  -> 16
  // (1,1): Ch0: 5*1+6*0+8*0+9*1=14, Ch1: 1*1+1*1+1*1+1*1=4  -> 18
  Tensor input = tf_int16.make(
      {1, 2, 3, 3},
      {1,
       2,
       3,
       4,
       5,
       6,
       7,
       8,
       9, // Channel 0
       1,
       1,
       1,
       1,
       1,
       1,
       1,
       1,
       1}); // Channel 1
  Tensor weight = tf_int8.make(
      {1, 2, 2, 2},
      {1,
       0,
       0,
       1, // Out Ch 0, In Ch 0: diagonal pattern
       1,
       1,
       1,
       1}); // Out Ch 0, In Ch 1: all ones
  Tensor bias = tf_int32.zeros({1});

  // Output dimensions: (3-2)/1+1=2 for each spatial dimension
  Tensor output = tf_int16.zeros({1, 1, 2, 2});

  int64_t in_zero_point = 0;
  Tensor weight_zero_point = tf_int32.make({1}, {0});
  Tensor bias_scale = tf_float.make({1}, {1.0f});
  double output_scale = 1.0;
  int64_t output_zero_point = 0;
  Tensor out_multiplier = tf_int32.make({1}, {1073741824}); // 0.5 * 2^31
  Tensor out_shift = tf_int32.make({1}, {0});

  std::array<int64_t, 2> stride_arr = {1, 1};
  std::array<int64_t, 2> padding_arr = {0, 0};
  std::array<int64_t, 2> dilation_arr = {1, 1};

  ::executorch::aten::ArrayRef<int64_t> stride(stride_arr.data(), 2);
  ::executorch::aten::ArrayRef<int64_t> padding(padding_arr.data(), 2);
  ::executorch::aten::ArrayRef<int64_t> dilation(dilation_arr.data(), 2);

  quantized_conv2d_nchw_out(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      1, // groups
      in_zero_point,
      weight_zero_point,
      bias_scale,
      output_scale,
      output_zero_point,
      out_multiplier,
      out_shift,
      output);

  Tensor expected = tf_int16.make({1, 1, 2, 2}, {10, 12, 16, 18});
  EXPECT_TENSOR_EQ(output, expected);
}

// Test quantized_conv2d_nhwc_out with int16 activations and int8 weights
TEST_F(HiFiQuantizedConv2dTest, QuantizedConv2dNhwcInt16Test) {
  TensorFactory<ScalarType::Short> tf_int16;
  TensorFactory<ScalarType::Int> tf_int32;
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;

  // Minimal test case in NHWC format: input [1, 3, 3, 2], kernel [1, 2, 2, 2]
  // -> output [1, 2, 2, 1] Same values as NCHW test, just different layout
  //
  // Input (H=3, W=3, C=2):
  // Position (h,w): [Ch0, Ch1]
  // (0,0): [1, 1]  (0,1): [2, 1]  (0,2): [3, 1]
  // (1,0): [4, 1]  (1,1): [5, 1]  (1,2): [6, 1]
  // (2,0): [7, 1]  (2,1): [8, 1]  (2,2): [9, 1]
  //
  // Weight (Out=1, H=2, W=2, In=2):
  // For output channel 0:
  // Position (h,w): [In0, In1]
  // (0,0): [1, 1]  (0,1): [0, 1]
  // (1,0): [0, 1]  (1,1): [1, 1]
  //
  // Hand calculation matches NCHW test:
  // Output (0,0): 10, (0,1): 12, (1,0): 16, (1,1): 18
  Tensor input = tf_int16.make(
      {1, 3, 3, 2},
      {1,
       1,
       2,
       1,
       3,
       1, // Row 0: (Ch0,Ch1) pairs
       4,
       1,
       5,
       1,
       6,
       1, // Row 1
       7,
       1,
       8,
       1,
       9,
       1}); // Row 2
  Tensor weight = tf_int8.make(
      {1, 2, 2, 2},
      {1,
       1,
       0,
       1, // Row 0: (In0,In1) pairs
       0,
       1,
       1,
       1}); // Row 1
  Tensor bias = tf_int32.zeros({1});

  // Output dimensions: (3-2)/1+1=2 for each spatial dimension
  Tensor output = tf_int16.zeros({1, 2, 2, 1});

  int64_t in_zero_point = 0;
  Tensor weight_zero_point = tf_int32.make({1}, {0});
  Tensor bias_scale = tf_float.make({1}, {1.0f});
  double output_scale = 1.0;
  int64_t output_zero_point = 0;
  Tensor out_multiplier = tf_int32.make({1}, {1073741824}); // 0.5 * 2^31
  Tensor out_shift = tf_int32.make({1}, {0});

  std::array<int64_t, 2> stride_arr = {1, 1};
  std::array<int64_t, 2> padding_arr = {0, 0};
  std::array<int64_t, 2> dilation_arr = {1, 1};

  ::executorch::aten::ArrayRef<int64_t> stride(stride_arr.data(), 2);
  ::executorch::aten::ArrayRef<int64_t> padding(padding_arr.data(), 2);
  ::executorch::aten::ArrayRef<int64_t> dilation(dilation_arr.data(), 2);

  quantized_conv2d_nhwc_out(
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      1, // groups
      in_zero_point,
      weight_zero_point,
      bias_scale,
      output_scale,
      output_zero_point,
      out_multiplier,
      out_shift,
      output);

  Tensor expected = tf_int16.make({1, 2, 2, 1}, {10, 12, 16, 18});
  EXPECT_TENSOR_EQ(output, expected);
}

} // namespace
} // namespace native
} // namespace HiFi
} // namespace impl
