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
};

// Test basic 3x3 kernel with NCHW layout
TEST_F(HiFiIm2rowTest, Basic3x3Kernel) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Int> tf_int;
  
  // Input shape: (1, 8, 5, 4) - batch, channels, height, width
  const std::vector<int32_t> input_sizes{1, 8, 5, 4};
  const int64_t kernel_size[] = {3, 3};
  const int64_t dilation[] = {1, 1};
  const int64_t padding[] = {0, 0};
  const int64_t stride[] = {1, 1};
  const bool channel_last = false;
  
  // Calculate output dimensions
  // out_h = (in_h + 2*pad_h - dilation_h*(kernel_h-1) - 1) / stride_h + 1
  //       = (5 + 0 - 1*2 - 1) / 1 + 1 = 3
  // out_w = (4 + 0 - 1*2 - 1) / 1 + 1 = 2
  // output_shape: (batch, out_h*out_w, kernel_h*kernel_w*channels)
  //             = (1, 6, 72)
  const std::vector<int32_t> output_sizes{1, 6, 72};
  
  Tensor input = tf.ones(input_sizes);
  Tensor zero_point = tf_int.zeros({1});
  Tensor out = tf.zeros(output_sizes);
  
  im2row_out(input, kernel_size, dilation, padding, stride, 
             zero_point, channel_last, out);
  
  // Print ALL output values to check for zero output error
  const float* out_data = out.const_data_ptr<float>();
  std::cout << "\n=== Basic3x3Kernel Output (all " << out.numel() << " elements) ===" << std::endl;
  for (int i = 0; i < out.numel(); ++i) {
    std::cout << out_data[i] << " ";
    if ((i + 1) % 10 == 0) std::cout << std::endl;  // New line every 10 elements
  }
  std::cout << std::endl;
  
  // Verify output has NO zeros at all
  int zero_count = 0;
  for (int i = 0; i < out.numel(); ++i) {
    if (out_data[i] == 0.0f) {
      zero_count++;
      if (zero_count <= 10) {
        std::cout << "ZERO found at index " << i << std::endl;
      }
    }
  }
  std::cout << "Total zeros found: " << zero_count << " out of " << out.numel() << " elements" << std::endl;
  EXPECT_EQ(zero_count, 0) << "Output should have NO zeros, but found " << zero_count << " zeros";
}

// Test with stride=2
TEST_F(HiFiIm2rowTest, WithStride2) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Int> tf_int;
  
  const std::vector<int32_t> input_sizes{1, 8, 5, 4};
  const int64_t kernel_size[] = {3, 3};
  const int64_t dilation[] = {1, 1};
  const int64_t padding[] = {0, 0};
  const int64_t stride[] = {2, 2};
  const bool channel_last = false;
  
  // out_h = (5 + 0 - 1*2 - 1) / 2 + 1 = 2
  // out_w = (4 + 0 - 1*2 - 1) / 2 + 1 = 1
  const std::vector<int32_t> output_sizes{1, 2, 72};
  
  Tensor input = tf.ones(input_sizes);
  Tensor zero_point = tf_int.zeros({1});
  Tensor out = tf.zeros(output_sizes);
  
  im2row_out(input, kernel_size, dilation, padding, stride,
             zero_point, channel_last, out);
  
  // Print ALL output values
  const float* out_data = out.const_data_ptr<float>();
  std::cout << "\n=== WithStride2 Output (all " << out.numel() << " elements) ===" << std::endl;
  for (int i = 0; i < out.numel(); ++i) {
    std::cout << out_data[i] << " ";
    if ((i + 1) % 10 == 0) std::cout << std::endl;
  }
  std::cout << std::endl;
  
  // Verify output has NO zeros at all
  int zero_count = 0;
  for (int i = 0; i < out.numel(); ++i) {
    if (out_data[i] == 0.0f) {
      zero_count++;
      if (zero_count <= 10) {
        std::cout << "ZERO found at index " << i << std::endl;
      }
    }
  }
  std::cout << "Total zeros found: " << zero_count << " out of " << out.numel() << " elements" << std::endl;
  EXPECT_EQ(zero_count, 0) << "Output should have NO zeros, but found " << zero_count << " zeros";
}

// Test with padding
TEST_F(HiFiIm2rowTest, WithPadding) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Int> tf_int;
  
  const std::vector<int32_t> input_sizes{1, 8, 5, 4};
  const int64_t kernel_size[] = {3, 3};
  const int64_t dilation[] = {1, 1};
  const int64_t padding[] = {1, 1};
  const int64_t stride[] = {1, 1};
  const bool channel_last = false;
  
  // out_h = (5 + 2*1 - 1*2 - 1) / 1 + 1 = 5
  // out_w = (4 + 2*1 - 1*2 - 1) / 1 + 1 = 4
  const std::vector<int32_t> output_sizes{1, 20, 72};
  
  Tensor input = tf.ones(input_sizes);
  Tensor zero_point = tf_int.zeros({1});
  Tensor out = tf.zeros(output_sizes);
  
  im2row_out(input, kernel_size, dilation, padding, stride,
             zero_point, channel_last, out);
  
  // Print ALL output values
  const float* out_data = out.const_data_ptr<float>();
  std::cout << "\n=== WithPadding Output (all " << out.numel() << " elements) ===" << std::endl;
  for (int i = 0; i < out.numel(); ++i) {
    std::cout << out_data[i] << " ";
    if ((i + 1) % 10 == 0) std::cout << std::endl;
  }
  std::cout << std::endl;
  
  // Verify output has NO zeros at all
  int zero_count = 0;
  for (int i = 0; i < out.numel(); ++i) {
    if (out_data[i] == 0.0f) {
      zero_count++;
      if (zero_count <= 10) {
        std::cout << "ZERO found at index " << i << std::endl;
      }
    }
  }
  std::cout << "Total zeros found: " << zero_count << " out of " << out.numel() << " elements" << std::endl;
  EXPECT_EQ(zero_count, 0) << "Output should have NO zeros, but found " << zero_count << " zeros";
}

// Test channels last (NHWC) layout
TEST_F(HiFiIm2rowTest, ChannelsLast) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Int> tf_int;
  
  // Input shape for NHWC: (1, 5, 8, 8) - batch, height, width, channels
  const std::vector<int32_t> input_sizes{1, 5, 8, 8};
  const int64_t kernel_size[] = {3, 3};
  const int64_t dilation[] = {1, 1};
  const int64_t padding[] = {0, 0};
  const int64_t stride[] = {1, 1};
  const bool channel_last = true;
  
  // out_h = (5 + 0 - 1*2 - 1) / 1 + 1 = 3
  // out_w = (8 + 0 - 1*2 - 1) / 1 + 1 = 6
  // channels = 8
  const std::vector<int32_t> output_sizes{1, 18, 72};
  
  Tensor input = tf.ones(input_sizes);
  Tensor zero_point = tf_int.zeros({1});
  Tensor out = tf.zeros(output_sizes);
  
  im2row_out(input, kernel_size, dilation, padding, stride,
             zero_point, channel_last, out);
  
  // Print ALL output values
  const float* out_data = out.const_data_ptr<float>();
  std::cout << "\n=== ChannelsLast Output (all " << out.numel() << " elements) ===" << std::endl;
  for (int i = 0; i < out.numel(); ++i) {
    std::cout << out_data[i] << " ";
    if ((i + 1) % 10 == 0) std::cout << std::endl;
  }
  std::cout << std::endl;
  
  // Verify output has NO zeros at all
  int zero_count = 0;
  for (int i = 0; i < out.numel(); ++i) {
    if (out_data[i] == 0.0f) {
      zero_count++;
      if (zero_count <= 10) {
        std::cout << "ZERO found at index " << i << std::endl;
      }
    }
  }
  std::cout << "Total zeros found: " << zero_count << " out of " << out.numel() << " elements" << std::endl;
  EXPECT_EQ(zero_count, 0) << "Output should have NO zeros, but found " << zero_count << " zeros";
}

// Test with dilation
TEST_F(HiFiIm2rowTest, WithDilation) {
  TensorFactory<ScalarType::Float> tf;
  TensorFactory<ScalarType::Int> tf_int;
  
  const std::vector<int32_t> input_sizes{1, 8, 6, 5};
  const int64_t kernel_size[] = {3, 3};
  const int64_t dilation[] = {2, 2};
  const int64_t padding[] = {2, 2};
  const int64_t stride[] = {1, 1};
  const bool channel_last = false;
  
  // out_h = (6 + 2*2 - 2*2 - 1) / 1 + 1 = 6
  // out_w = (5 + 2*2 - 2*2 - 1) / 1 + 1 = 5
  const std::vector<int32_t> output_sizes{1, 30, 72};
  
  Tensor input = tf.ones(input_sizes);
  Tensor zero_point = tf_int.zeros({1});
  Tensor out = tf.zeros(output_sizes);
  
  im2row_out(input, kernel_size, dilation, padding, stride,
             zero_point, channel_last, out);
  
  // Print ALL output values
  const float* out_data = out.const_data_ptr<float>();
  std::cout << "\n=== WithDilation Output (all " << out.numel() << " elements) ===" << std::endl;
  for (int i = 0; i < out.numel(); ++i) {
    std::cout << out_data[i] << " ";
    if ((i + 1) % 10 == 0) std::cout << std::endl;
  }
  std::cout << std::endl;
  
  // Verify output has NO zeros at all
  int zero_count = 0;
  for (int i = 0; i < out.numel(); ++i) {
    if (out_data[i] == 0.0f) {
      zero_count++;
      if (zero_count <= 10) {
        std::cout << "ZERO found at index " << i << std::endl;
      }
    }
  }
  std::cout << "Total zeros found: " << zero_count << " out of " << out.numel() << " elements" << std::endl;
  EXPECT_EQ(zero_count, 0) << "Output should have NO zeros, but found " << zero_count << " zeros";
}

// Test im2row_per_tensor_out with zero_point = 0
TEST_F(HiFiIm2rowTest, PerTensorZeroPointZero) {
  TensorFactory<ScalarType::Float> tf;
  
  const std::vector<int32_t> input_sizes{1, 8, 5, 4};
  const int64_t kernel_size[] = {3, 3};
  const int64_t dilation[] = {1, 1};
  const int64_t padding[] = {0, 0};
  const int64_t stride[] = {1, 1};
  const int64_t in_zero_point = 0;
  const bool channel_last = false;
  
  const std::vector<int32_t> output_sizes{1, 6, 72};
  
  Tensor input = tf.ones(input_sizes);
  Tensor out = tf.zeros(output_sizes);
  
  im2row_per_tensor_out(input, kernel_size, dilation, padding, stride,
                        in_zero_point, channel_last, out);
  
  // Print ALL output values
  const float* out_data = out.const_data_ptr<float>();
  std::cout << "\n=== PerTensorZeroPointZero Output (all " << out.numel() << " elements) ===" << std::endl;
  for (int i = 0; i < out.numel(); ++i) {
    std::cout << out_data[i] << " ";
    if ((i + 1) % 10 == 0) std::cout << std::endl;
  }
  std::cout << std::endl;
  
  // Verify output has NO zeros at all
  int zero_count = 0;
  for (int i = 0; i < out.numel(); ++i) {
    if (out_data[i] == 0.0f) {
      zero_count++;
      if (zero_count <= 10) {
        std::cout << "ZERO found at index " << i << std::endl;
      }
    }
  }
  std::cout << "Total zeros found: " << zero_count << " out of " << out.numel() << " elements" << std::endl;
  EXPECT_EQ(zero_count, 0) << "Output should have NO zeros, but found " << zero_count << " zeros";
}

// Test im2row_per_tensor_out with non-zero zero_point
TEST_F(HiFiIm2rowTest, PerTensorNonZeroZeroPoint) {
  TensorFactory<ScalarType::Float> tf;
  
  const std::vector<int32_t> input_sizes{1, 8, 5, 4};
  const int64_t kernel_size[] = {3, 3};
  const int64_t dilation[] = {1, 1};
  const int64_t padding[] = {0, 0};
  const int64_t stride[] = {1, 1};
  const int64_t in_zero_point = 128;
  const bool channel_last = false;
  
  const std::vector<int32_t> output_sizes{1, 6, 72};
  
  Tensor input = tf.ones(input_sizes);
  Tensor out = tf.zeros(output_sizes);
  
  im2row_per_tensor_out(input, kernel_size, dilation, padding, stride,
                        in_zero_point, channel_last, out);
  
  // Print ALL output values
  const float* out_data = out.const_data_ptr<float>();
  std::cout << "\n=== PerTensorNonZeroZeroPoint Output (all " << out.numel() << " elements) ===" << std::endl;
  for (int i = 0; i < out.numel(); ++i) {
    std::cout << out_data[i] << " ";
    if ((i + 1) % 10 == 0) std::cout << std::endl;
  }
  std::cout << std::endl;
  
  // Verify output has NO zeros at all
  int zero_count = 0;
  for (int i = 0; i < out.numel(); ++i) {
    if (out_data[i] == 0.0f) {
      zero_count++;
      if (zero_count <= 10) {
        std::cout << "ZERO found at index " << i << std::endl;
      }
    }
  }
  std::cout << "Total zeros found: " << zero_count << " out of " << out.numel() << " elements" << std::endl;
  EXPECT_EQ(zero_count, 0) << "Output should have NO zeros, but found " << zero_count << " zeros";
}

// Test im2row_per_tensor_out with channels last and non-zero zero_point
TEST_F(HiFiIm2rowTest, PerTensorChannelsLastNonZeroZeroPoint) {
  TensorFactory<ScalarType::Float> tf;
  
  const std::vector<int32_t> input_sizes{1, 5, 8, 8};
  const int64_t kernel_size[] = {3, 3};
  const int64_t dilation[] = {1, 1};
  const int64_t padding[] = {0, 0};
  const int64_t stride[] = {1, 1};
  const int64_t in_zero_point = 64;
  const bool channel_last = true;
  
  const std::vector<int32_t> output_sizes{1, 18, 72};
  
  Tensor input = tf.ones(input_sizes);
  Tensor out = tf.zeros(output_sizes);
  
  im2row_per_tensor_out(input, kernel_size, dilation, padding, stride,
                        in_zero_point, channel_last, out);
  
  // Print ALL output values
  const float* out_data = out.const_data_ptr<float>();
  std::cout << "\n=== PerTensorChannelsLastNonZeroZeroPoint Output (all " << out.numel() << " elements) ===" << std::endl;
  for (int i = 0; i < out.numel(); ++i) {
    std::cout << out_data[i] << " ";
    if ((i + 1) % 10 == 0) std::cout << std::endl;
  }
  std::cout << std::endl;
  
  // Verify output has NO zeros at all
  int zero_count = 0;
  for (int i = 0; i < out.numel(); ++i) {
    if (out_data[i] == 0.0f) {
      zero_count++;
      if (zero_count <= 10) {
        std::cout << "ZERO found at index " << i << std::endl;
      }
    }
  }
  std::cout << "Total zeros found: " << zero_count << " out of " << out.numel() << " elements" << std::endl;
  EXPECT_EQ(zero_count, 0) << "Output should have NO zeros, but found " << zero_count << " zeros";
}

} // namespace
} // namespace native
} // namespace HiFi
} // namespace impl
