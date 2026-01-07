/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <sys/times.h>
#include <xtensa/sim.h>

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

using ::executorch::aten::ArrayRef;
using ::executorch::aten::IntArrayRef;
using ::executorch::aten::Scalar;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::aten::TensorImpl;
using ::executorch::runtime::Error;
using ::executorch::runtime::KernelRuntimeContext;
using ::executorch::runtime::runtime_init;
using ::executorch::runtime::testing::TensorFactory;

class HiFiIm2rowTest : public OperatorTest {
 public:
 protected:
  void im2row_out(
      const Tensor& input,
      IntArrayRef kernel_size,
      IntArrayRef dilation,
      IntArrayRef padding,
      IntArrayRef stride,
      const Tensor& in_zero_point,
      bool channel_last,
      Tensor& out) {
    ::impl::HiFi::native::im2row_out(
        context_, input, kernel_size, dilation, padding, stride,
        in_zero_point, channel_last, out);
  }

  void im2row_per_tensor_out(
      const Tensor& input,
      IntArrayRef kernel_size,
      IntArrayRef dilation,
      IntArrayRef padding,
      IntArrayRef stride,
      int64_t in_zero_point,
      bool channel_last,
      Tensor& out) {
    ::impl::HiFi::native::im2row_per_tensor_out(
        context_, input, kernel_size, dilation, padding, stride,
        in_zero_point, channel_last, out);
  }

  // Helper to count occurrences of a value in output tensor
  int countValue(const Tensor& tensor, float value) {
    const float* data = tensor.const_data_ptr<float>();
    int count = 0;
    for (int i = 0; i < tensor.numel(); ++i) {
      if (data[i] == value) {
        count++;
      }
    }
    return count;
  }
};

// Test basic 3x3 kernel with NCHW layout, no padding
// Input is all ones, so output should be all ones
TEST_F(HiFiIm2rowTest, Basic3x3Kernel) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Int> tf_int;

  // Input: (1, 8, 5, 4) - batch=1, channels=8, height=5, width=4
  const std::vector<int32_t> input_sizes{1, 8, 5, 4};
  const int64_t kernel_size[] = {3, 3};
  const int64_t dilation[] = {1, 1};
  const int64_t padding[] = {0, 0};
  const int64_t stride[] = {1, 1};
  const bool channel_last = false;

  // out_h = (5 - 3) / 1 + 1 = 3
  // out_w = (4 - 3) / 1 + 1 = 2
  // output: (1, 3*2, 3*3*8) = (1, 6, 72)
  const std::vector<int32_t> output_sizes{1, 6, 72};

  Tensor input = tf.ones(input_sizes);
  Tensor zero_point = tf_int.zeros({1});
  Tensor out = tf.zeros(output_sizes);

  im2row_out(input, kernel_size, dilation, padding, stride,
             zero_point, channel_last, out);

  // Without padding, all output values should be 1.0 (from input)
  EXPECT_EQ(countValue(out, 1.0f), out.numel());
  EXPECT_EQ(countValue(out, 0.0f), 0);
}

// Test with stride=2, no padding
// Input is all ones, so output should be all ones
TEST_F(HiFiIm2rowTest, WithStride2) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Int> tf_int;

  const std::vector<int32_t> input_sizes{1, 8, 5, 4};
  const int64_t kernel_size[] = {3, 3};
  const int64_t dilation[] = {1, 1};
  const int64_t padding[] = {0, 0};
  const int64_t stride[] = {2, 2};
  const bool channel_last = false;

  // out_h = (5 - 3) / 2 + 1 = 2
  // out_w = (4 - 3) / 2 + 1 = 1
  // output: (1, 2*1, 3*3*8) = (1, 2, 72)
  const std::vector<int32_t> output_sizes{1, 2, 72};

  Tensor input = tf.ones(input_sizes);
  Tensor zero_point = tf_int.zeros({1});
  Tensor out = tf.zeros(output_sizes);

  im2row_out(input, kernel_size, dilation, padding, stride,
             zero_point, channel_last, out);

  // Without padding, all output values should be 1.0 (from input)
  EXPECT_EQ(countValue(out, 1.0f), out.numel());
  EXPECT_EQ(countValue(out, 0.0f), 0);
}

// Test with padding=1
// With zero_point=0, padded regions produce zeros in output
TEST_F(HiFiIm2rowTest, WithPadding) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Int> tf_int;

  const std::vector<int32_t> input_sizes{1, 8, 5, 4};
  const int64_t kernel_size[] = {3, 3};
  const int64_t dilation[] = {1, 1};
  const int64_t padding[] = {1, 1};
  const int64_t stride[] = {1, 1};
  const bool channel_last = false;

  // out_h = (5 + 2*1 - 3) / 1 + 1 = 5
  // out_w = (4 + 2*1 - 3) / 1 + 1 = 4
  // output: (1, 5*4, 3*3*8) = (1, 20, 72)
  const std::vector<int32_t> output_sizes{1, 20, 72};

  Tensor input = tf.ones(input_sizes);
  Tensor zero_point = tf_int.zeros({1});
  Tensor out = tf.zeros(output_sizes);

  im2row_out(input, kernel_size, dilation, padding, stride,
             zero_point, channel_last, out);

  int one_count = countValue(out, 1.0f);
  int zero_count = countValue(out, 0.0f);

  // With padding and zero_point=0: expect both ones (from input) and zeros (from padding)
  EXPECT_GT(one_count, 0) << "Should have ones from input data";
  EXPECT_GT(zero_count, 0) << "Should have zeros from padded regions";
  EXPECT_EQ(one_count + zero_count, out.numel()) << "All values should be 0 or 1";
}

// Test channels last (NHWC) layout, no padding
// Input is all ones, so output should be all ones
TEST_F(HiFiIm2rowTest, ChannelsLast) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Int> tf_int;

  // Input for NHWC: (1, 5, 8, 8) - batch=1, height=5, width=8, channels=8
  const std::vector<int32_t> input_sizes{1, 5, 8, 8};
  const int64_t kernel_size[] = {3, 3};
  const int64_t dilation[] = {1, 1};
  const int64_t padding[] = {0, 0};
  const int64_t stride[] = {1, 1};
  const bool channel_last = true;

  // out_h = (5 - 3) / 1 + 1 = 3
  // out_w = (8 - 3) / 1 + 1 = 6
  // output: (1, 3*6, 3*3*8) = (1, 18, 72)
  const std::vector<int32_t> output_sizes{1, 18, 72};

  Tensor input = tf.ones(input_sizes);
  Tensor zero_point = tf_int.zeros({1});
  Tensor out = tf.zeros(output_sizes);

  im2row_out(input, kernel_size, dilation, padding, stride,
             zero_point, channel_last, out);

  // Without padding, all output values should be 1.0 (from input)
  EXPECT_EQ(countValue(out, 1.0f), out.numel());
  EXPECT_EQ(countValue(out, 0.0f), 0);
}

// Test with dilation=2 and padding=2
// With zero_point=0, dilated regions outside input produce zeros
TEST_F(HiFiIm2rowTest, WithDilation) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Int> tf_int;

  const std::vector<int32_t> input_sizes{1, 8, 6, 5};
  const int64_t kernel_size[] = {3, 3};
  const int64_t dilation[] = {2, 2};
  const int64_t padding[] = {2, 2};
  const int64_t stride[] = {1, 1};
  const bool channel_last = false;

  // effective_kernel_h = 2*(3-1) + 1 = 5
  // effective_kernel_w = 2*(3-1) + 1 = 5
  // out_h = (6 + 2*2 - 5) / 1 + 1 = 6
  // out_w = (5 + 2*2 - 5) / 1 + 1 = 5
  // output: (1, 6*5, 3*3*8) = (1, 30, 72)
  const std::vector<int32_t> output_sizes{1, 30, 72};

  Tensor input = tf.ones(input_sizes);
  Tensor zero_point = tf_int.zeros({1});
  Tensor out = tf.zeros(output_sizes);

  im2row_out(input, kernel_size, dilation, padding, stride,
             zero_point, channel_last, out);

  int one_count = countValue(out, 1.0f);
  int zero_count = countValue(out, 0.0f);

  // With dilation/padding and zero_point=0: expect both ones and zeros
  EXPECT_GT(one_count, 0) << "Should have ones from input data";
  EXPECT_GT(zero_count, 0) << "Should have zeros from padded/dilated regions";
  EXPECT_EQ(one_count + zero_count, out.numel()) << "All values should be 0 or 1";
}

// Test im2row_per_tensor_out with zero_point=0, no padding
// Input is all ones, so output should be all ones
TEST_F(HiFiIm2rowTest, PerTensorZeroPointZero) {
  TensorFactory<ScalarType::Float> tf;

  const std::vector<int32_t> input_sizes{1, 8, 5, 4};
  const int64_t kernel_size[] = {3, 3};
  const int64_t dilation[] = {1, 1};
  const int64_t padding[] = {0, 0};
  const int64_t stride[] = {1, 1};
  const int64_t in_zero_point = 0;
  const bool channel_last = false;

  // output: (1, 6, 72)
  const std::vector<int32_t> output_sizes{1, 6, 72};

  Tensor input = tf.ones(input_sizes);
  Tensor out = tf.zeros(output_sizes);

  im2row_per_tensor_out(input, kernel_size, dilation, padding, stride,
                        in_zero_point, channel_last, out);

  // Without padding, all output values should be 1.0 (from input)
  EXPECT_EQ(countValue(out, 1.0f), out.numel());
  EXPECT_EQ(countValue(out, 0.0f), 0);
}

// Test im2row_per_tensor_out with non-zero zero_point=128, no padding
// Input is all ones, so output should be all ones
TEST_F(HiFiIm2rowTest, PerTensorNonZeroZeroPoint) {
  TensorFactory<ScalarType::Float> tf;

  const std::vector<int32_t> input_sizes{1, 8, 5, 4};
  const int64_t kernel_size[] = {3, 3};
  const int64_t dilation[] = {1, 1};
  const int64_t padding[] = {0, 0};
  const int64_t stride[] = {1, 1};
  const int64_t in_zero_point = 128;
  const bool channel_last = false;

  // output: (1, 6, 72)
  const std::vector<int32_t> output_sizes{1, 6, 72};

  Tensor input = tf.ones(input_sizes);
  Tensor out = tf.zeros(output_sizes);

  im2row_per_tensor_out(input, kernel_size, dilation, padding, stride,
                        in_zero_point, channel_last, out);

  // Without padding, all output values should be 1.0 (from input)
  // zero_point only affects padded regions, which don't exist here
  EXPECT_EQ(countValue(out, 1.0f), out.numel());
  EXPECT_EQ(countValue(out, 0.0f), 0);
}

// Test im2row_per_tensor_out with channels last layout and non-zero zero_point
// Input is all ones, so output should be all ones
TEST_F(HiFiIm2rowTest, PerTensorChannelsLastNonZeroZeroPoint) {
  TensorFactory<ScalarType::Float> tf;

  const std::vector<int32_t> input_sizes{1, 5, 8, 8};
  const int64_t kernel_size[] = {3, 3};
  const int64_t dilation[] = {1, 1};
  const int64_t padding[] = {0, 0};
  const int64_t stride[] = {1, 1};
  const int64_t in_zero_point = 64;
  const bool channel_last = true;

  // output: (1, 18, 72)
  const std::vector<int32_t> output_sizes{1, 18, 72};

  Tensor input = tf.ones(input_sizes);
  Tensor out = tf.zeros(output_sizes);

  im2row_per_tensor_out(input, kernel_size, dilation, padding, stride,
                        in_zero_point, channel_last, out);

  // Without padding, all output values should be 1.0 (from input)
  EXPECT_EQ(countValue(out, 1.0f), out.numel());
  EXPECT_EQ(countValue(out, 0.0f), 0);
}

} // namespace
} // namespace native
} // namespace HiFi
} // namespace impl
