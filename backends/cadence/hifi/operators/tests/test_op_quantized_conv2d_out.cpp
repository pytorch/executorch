/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <sys/times.h>

#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <executorch/runtime/platform/runtime.h>

#include <executorch/backends/cadence/hifi/operators/operators.h>

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
using std::optional;
using std::string_view;

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

  // Simple 2D case: input [1, 8, 20, 28] with kernel [16, 8, 3, 5]
  // Using simple values for testing
  Tensor input = tf_int16.ones({1, 8, 20, 28});
  Tensor weight = tf_int8.ones({16, 8, 3, 5});
  Tensor bias = tf_int32.zeros({16});

  // Calculate output dimensions: (20-3)/1+1=18, (28-5)/1+1=24
  Tensor output = tf_int16.zeros({1, 16, 18, 24});

  int64_t in_zero_point = 0;
  Tensor weight_zero_point = tf_int32.make({1}, {0});
  Tensor bias_scale = tf_float.make({1}, {1.0f});
  double output_scale = 1.0;
  int64_t output_zero_point = 0;
  Tensor out_multiplier = tf_int32.make({1}, {1073741824}); // 0.5 * 2^31
  Tensor out_shift = tf_int32.make({1}, {0});

  int64_t stride_arr[] = {1, 1};
  int64_t padding_arr[] = {0, 0};
  int64_t dilation_arr[] = {1, 1};

  ::executorch::aten::ArrayRef<int64_t> stride(stride_arr, 2);
  ::executorch::aten::ArrayRef<int64_t> padding(padding_arr, 2);
  ::executorch::aten::ArrayRef<int64_t> dilation(dilation_arr, 2);

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

  // Basic sanity check - output should be non-zero
  // With all ones input and weights, and kernel size 3x5=15 * 8 channels = 120
  // Expected value per output element would be around 120
  EXPECT_NE(output.const_data_ptr<int16_t>()[0], 0);
}

// Test quantized_conv2d_nhwc_out with int16 activations and int8 weights
TEST_F(HiFiQuantizedConv2dTest, QuantizedConv2dNhwcInt16Test) {
  TensorFactory<ScalarType::Short> tf_int16;
  TensorFactory<ScalarType::Int> tf_int32;
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Float> tf_float;

  // Simple 2D case in NHWC format: input [1, 20, 28, 8] with kernel [16, 3, 5,
  // 8]
  Tensor input = tf_int16.ones({1, 20, 28, 8});
  Tensor weight = tf_int8.ones({16, 3, 5, 8});
  Tensor bias = tf_int32.zeros({16});

  // Calculate output dimensions: (20-3)/1+1=18, (28-5)/1+1=24
  Tensor output = tf_int16.zeros({1, 18, 24, 16});

  int64_t in_zero_point = 0;
  Tensor weight_zero_point = tf_int32.make({1}, {0});
  Tensor bias_scale = tf_float.make({1}, {1.0f});
  double output_scale = 1.0;
  int64_t output_zero_point = 0;
  Tensor out_multiplier = tf_int32.make({1}, {1073741824}); // 0.5 * 2^31
  Tensor out_shift = tf_int32.make({1}, {0});

  int64_t stride_arr[] = {1, 1};
  int64_t padding_arr[] = {0, 0};
  int64_t dilation_arr[] = {1, 1};

  ::executorch::aten::ArrayRef<int64_t> stride(stride_arr, 2);
  ::executorch::aten::ArrayRef<int64_t> padding(padding_arr, 2);
  ::executorch::aten::ArrayRef<int64_t> dilation(dilation_arr, 2);

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

  // Basic sanity check - output should be non-zero
  EXPECT_NE(output.const_data_ptr<int16_t>()[0], 0);
}

} // namespace
} // namespace native
} // namespace HiFi
} // namespace impl
