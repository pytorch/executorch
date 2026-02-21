/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_quantized_softmax.h>

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
using ::executorch::runtime::KernelRuntimeContext;
using ::executorch::runtime::testing::TensorFactory;

class GenericQuantizedSoftmaxTest : public OperatorTest {
 public:
 protected:
  // Helper that accepts explicit mask_type value
  Tensor& quantized_softmax_per_tensor_out(
      const Tensor& input,
      const Tensor& mask,
      int64_t dim,
      int64_t mask_type,
      const Tensor& pos,
      double in_scale,
      int64_t in_zero_point,
      double out_scale,
      int64_t out_zero_point,
      Tensor& output) {
    return impl::generic::native::quantized_softmax_per_tensor_out(
        context_,
        input,
        mask,
        dim,
        mask_type,
        pos,
        in_scale,
        in_zero_point,
        out_scale,
        out_zero_point,
        output);
  }
};

// Test basic softmax without masking (mask_type = 0)
// Uses a 4x16 input tensor with explicit data values
TEST_F(GenericQuantizedSoftmaxTest, BasicSoftmaxInt8NoMask) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Long> tf_int64;

  // ============================================================
  // Tensor dimensions
  // ============================================================
  constexpr int kNumRows = 4;
  constexpr int kNumCols = 16;

  // ============================================================
  // Input tensor: 4 rows x 16 cols (64 elements)
  // Row 0: values 10-25 (dequantized: 1.0 to 2.5)
  // Row 1: values 20-35 (dequantized: 2.0 to 3.5)
  // Row 2: uniform values (dequantized: all 5.0)
  // Row 3: alternating pattern
  // ============================================================
  // clang-format off
  Tensor input = tf_int8.make(
      {kNumRows, kNumCols},
      {
          // Row 0
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
          // Row 1
          20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
          // Row 2
          50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
          // Row 3
          10, 30, 10, 30, 10, 30, 10, 30, 10, 30, 10, 30, 10, 30, 10, 30
      });
  // clang-format on

  // ============================================================
  // Output tensor: 4 rows x 16 cols, initialized to zeros
  // ============================================================
  Tensor output = tf_int8.zeros({kNumRows, kNumCols});

  // ============================================================
  // Mask tensor: single element (unused when mask_type = 0)
  // ============================================================
  Tensor mask = tf_int8.make({1}, {0});

  // ============================================================
  // Position tensor: single element (unused when mask_type = 0)
  // ============================================================
  Tensor pos = tf_int64.make({1}, {0});

  // ============================================================
  // Quantization parameters
  // ============================================================
  const double in_scale = 0.1;
  const int64_t in_zero_point = 0;
  const double out_scale = 1.0 / 127.0;
  const int64_t out_zero_point = 0;

  // ============================================================
  // Expected output tensor
  // Softmax is computed on dequantized values, then requantized.
  // Row 0: dequantized [1.0, 1.1, ..., 2.5], softmax applied
  // Row 1: dequantized [2.0, 2.1, ..., 3.5], softmax applied
  // Row 2: dequantized [5.0, 5.0, ..., 5.0], uniform -> ~8 each (127/16)
  // Row 3: dequantized alternating [1.0, 3.0, ...], bimodal distribution
  // ============================================================
  // clang-format off
  Tensor expected = tf_int8.make(
      {kNumRows, kNumCols},
      {
          // Row 0: increasing values [1.0-2.5], softmax gives increasing probs
          3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 11, 12, 14, 15,
          // Row 1: increasing values [2.0-3.5], softmax gives increasing probs
          3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 11, 12, 14, 15,
          // Row 2: uniform input -> uniform output (~8 each, 127/16 ≈ 7.9)
          8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
          // Row 3: alternating [1.0, 3.0] -> bimodal [low, high, low, high, ...]
          2, 14, 2, 14, 2, 14, 2, 14, 2, 14, 2, 14, 2, 14, 2, 14
      });
  // clang-format on

  // ============================================================
  // Execute softmax
  // ============================================================
  quantized_softmax_per_tensor_out(
      input,
      mask,
      -1, // dim = last dimension
      0, // mask_type = 0 (no masking)
      pos,
      in_scale,
      in_zero_point,
      out_scale,
      out_zero_point,
      output);

  EXPECT_TENSOR_EQ(output, expected);
}

// Test softmax with uint8 input type using larger tensor
TEST_F(GenericQuantizedSoftmaxTest, BasicSoftmaxUInt8NoMask) {
  TensorFactory<ScalarType::Byte> tf_uint8;
  TensorFactory<ScalarType::Long> tf_int64;

  // ============================================================
  // Tensor dimensions
  // ============================================================
  constexpr int kNumRows = 8;
  constexpr int kNumCols = 32;

  // ============================================================
  // Input tensor: 8 rows x 32 cols (256 elements)
  // Pattern: (i % 32) + 10 for each element
  // ============================================================
  // clang-format off
  Tensor input = tf_uint8.make(
      {kNumRows, kNumCols},
      {
          // Row 0
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
          // Row 1
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
          // Row 2
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
          // Row 3
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
          // Row 4
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
          // Row 5
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
          // Row 6
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
          // Row 7
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41
      });
  // clang-format on

  // ============================================================
  // Output tensor: 8 rows x 32 cols, initialized to zeros
  // ============================================================
  Tensor output = tf_uint8.zeros({kNumRows, kNumCols});

  // ============================================================
  // Mask tensor: single element (unused)
  // ============================================================
  Tensor mask = tf_uint8.make({1}, {0});

  // ============================================================
  // Position tensor: single element (unused)
  // ============================================================
  Tensor pos = tf_int64.make({1}, {0});

  // ============================================================
  // Quantization parameters
  // ============================================================
  const double in_scale = 0.1;
  const int64_t in_zero_point = 0;
  const double out_scale = 1.0 / 255.0;
  const int64_t out_zero_point = 0;

  // ============================================================
  // Expected output tensor
  // All rows have the same increasing pattern, softmax distribution
  // ============================================================
  // clang-format off
  Tensor expected = tf_uint8.make(
      {kNumRows, kNumCols},
      {
          // Row 0: increasing values [1.0-4.1], softmax distribution
          1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 11, 13, 14, 15, 17, 19, 21, 23, 25,
          // Row 1
          1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 11, 13, 14, 15, 17, 19, 21, 23, 25,
          // Row 2
          1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 11, 13, 14, 15, 17, 19, 21, 23, 25,
          // Row 3
          1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 11, 13, 14, 15, 17, 19, 21, 23, 25,
          // Row 4
          1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 11, 13, 14, 15, 17, 19, 21, 23, 25,
          // Row 5
          1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 11, 13, 14, 15, 17, 19, 21, 23, 25,
          // Row 6
          1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 11, 13, 14, 15, 17, 19, 21, 23, 25,
          // Row 7
          1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 11, 13, 14, 15, 17, 19, 21, 23, 25
      });
  // clang-format on

  // ============================================================
  // Execute softmax
  // ============================================================
  quantized_softmax_per_tensor_out(
      input,
      mask,
      -1,
      0, // mask_type = 0
      pos,
      in_scale,
      in_zero_point,
      out_scale,
      out_zero_point,
      output);

  EXPECT_TENSOR_EQ(output, expected);
}

// Test softmax with int16 input type
TEST_F(GenericQuantizedSoftmaxTest, BasicSoftmaxInt16NoMask) {
  TensorFactory<ScalarType::Short> tf_int16;
  TensorFactory<ScalarType::Long> tf_int64;

  // ============================================================
  // Tensor dimensions
  // ============================================================
  constexpr int kNumRows = 4;
  constexpr int kNumCols = 16;

  // ============================================================
  // Input tensor: 4 rows x 16 cols
  // Pattern: increasing values per row
  // ============================================================
  // clang-format off
  Tensor input = tf_int16.make(
      {kNumRows, kNumCols},
      {
          // Row 0: 100, 110, 120, ..., 250
          100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250,
          // Row 1
          100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250,
          // Row 2
          100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250,
          // Row 3
          100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250
      });
  // clang-format on

  // ============================================================
  // Output tensor: 4 rows x 16 cols, initialized to zeros
  // ============================================================
  Tensor output = tf_int16.zeros({kNumRows, kNumCols});

  // ============================================================
  // Mask tensor: single element (unused)
  // ============================================================
  Tensor mask = tf_int16.make({1}, {0});

  // ============================================================
  // Position tensor: single element (unused)
  // ============================================================
  Tensor pos = tf_int64.make({1}, {0});

  // ============================================================
  // Quantization parameters
  // ============================================================
  const double in_scale = 0.01;
  const int64_t in_zero_point = 0;
  const double out_scale = 1.0 / 32767.0;
  const int64_t out_zero_point = 0;

  // ============================================================
  // Execute softmax
  // ============================================================
  quantized_softmax_per_tensor_out(
      input,
      mask,
      -1,
      0,
      pos,
      in_scale,
      in_zero_point,
      out_scale,
      out_zero_point,
      output);

  // ============================================================
  // Expected output tensor
  // All rows have the same increasing pattern, softmax distribution
  // With scale=0.00003 and zero_point=0, values are in int16 range
  // ============================================================
  // clang-format off
  Tensor expected = tf_int16.make(
      {kNumRows, kNumCols},
      {
          // Row 0: increasing values [1.0-2.5], softmax distribution
          872, 963, 1065, 1177, 1301, 1437, 1588, 1756, 1940, 2144, 2370, 2619, 2894, 3199, 3535, 3907,
          // Row 1
          872, 963, 1065, 1177, 1301, 1437, 1588, 1756, 1940, 2144, 2370, 2619, 2894, 3199, 3535, 3907,
          // Row 2
          872, 963, 1065, 1177, 1301, 1437, 1588, 1756, 1940, 2144, 2370, 2619, 2894, 3199, 3535, 3907,
          // Row 3
          872, 963, 1065, 1177, 1301, 1437, 1588, 1756, 1940, 2144, 2370, 2619, 2894, 3199, 3535, 3907
      });
  // clang-format on

  EXPECT_TENSOR_EQ(output, expected);
}

// Test softmax with position-based causal masking (mask_type = 1)
TEST_F(GenericQuantizedSoftmaxTest, SoftmaxWithCausalMaskingInt8) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Long> tf_int64;

  // ============================================================
  // Tensor dimensions: 8 rows x 16 cols (simulating attention matrix)
  // ============================================================
  constexpr int kNumRows = 8;
  constexpr int kNumCols = 16;

  // ============================================================
  // Input tensor: attention scores
  // ============================================================
  // clang-format off
  Tensor input = tf_int8.make(
      {kNumRows, kNumCols},
      {
          // Row 0
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
          // Row 1
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
          // Row 2
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
          // Row 3
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
          // Row 4
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
          // Row 5
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
          // Row 6
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
          // Row 7
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
      });
  // clang-format on

  // ============================================================
  // Output tensor: 8 rows x 16 cols, initialized to zeros
  // ============================================================
  Tensor output = tf_int8.zeros({kNumRows, kNumCols});

  // ============================================================
  // Mask tensor: single element (unused when pos is used)
  // ============================================================
  Tensor mask = tf_int8.make({1}, {0});

  // ============================================================
  // Position tensor: start attending from position 3
  // ============================================================
  Tensor pos = tf_int64.make({1}, {3});

  // ============================================================
  // Quantization parameters
  // ============================================================
  const double in_scale = 0.1;
  const int64_t in_zero_point = 0;
  const double out_scale = 1.0 / 127.0;
  const int64_t out_zero_point = 0;

  // ============================================================
  // Execute softmax
  // ============================================================
  quantized_softmax_per_tensor_out(
      input,
      mask,
      -1,
      1, // mask_type = 1 (position-based causal masking)
      pos,
      in_scale,
      in_zero_point,
      out_scale,
      out_zero_point,
      output);

  // ============================================================
  // Expected output tensor
  // Causal masking with pos=3: row i attends to positions 0..(3+i)
  // Row 0 (pos=3): attend 0-3, mask 4-15
  // Row 1 (pos=4): attend 0-4, mask 5-15
  // Row 2 (pos=5): attend 0-5, mask 6-15
  // Row 3 (pos=6): attend 0-6, mask 7-15
  // Row 4 (pos=7): attend 0-7, mask 8-15
  // Row 5 (pos=8): attend 0-8, mask 9-15
  // Row 6 (pos=9): attend 0-9, mask 10-15
  // Row 7 (pos=10): attend 0-10, mask 11-15
  // ============================================================
  // clang-format off
  Tensor expected = tf_int8.make(
      {kNumRows, kNumCols},
      {
          // Row 0: attend cols 0-3, mask cols 4-15
          27, 30, 33, 37, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          // Row 1: attend cols 0-4, mask cols 5-15
          21, 23, 25, 28, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          // Row 2: attend cols 0-5, mask cols 6-15
          16, 18, 20, 22, 24, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          // Row 3: attend cols 0-6, mask cols 7-15
          13, 15, 16, 18, 20, 22, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          // Row 4: attend cols 0-7, mask cols 8-15
          11, 12, 13, 15, 16, 18, 20, 22, 0, 0, 0, 0, 0, 0, 0, 0,
          // Row 5: attend cols 0-8, mask cols 9-15
          9, 10, 11, 12, 14, 15, 17, 18, 20, 0, 0, 0, 0, 0, 0, 0,
          // Row 6: attend cols 0-9, mask cols 10-15
          8, 9, 9, 10, 12, 13, 14, 16, 17, 19, 0, 0, 0, 0, 0, 0,
          // Row 7: attend cols 0-10, mask cols 11-15
          7, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 0, 0, 0, 0, 0
      });
  // clang-format on

  EXPECT_TENSOR_EQ(output, expected);
}

// Test softmax with uniform input (all same values)
TEST_F(GenericQuantizedSoftmaxTest, SoftmaxUniformInputInt8) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Long> tf_int64;

  // ============================================================
  // Tensor dimensions
  // ============================================================
  constexpr int kNumRows = 4;
  constexpr int kNumCols = 16;

  // ============================================================
  // Input tensor: all same values (uniform) = 50
  // ============================================================
  // clang-format off
  Tensor input = tf_int8.make(
      {kNumRows, kNumCols},
      {
          // Row 0
          50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
          // Row 1
          50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
          // Row 2
          50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
          // Row 3
          50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50
      });
  // clang-format on

  // ============================================================
  // Output tensor: initialized to zeros
  // ============================================================
  Tensor output = tf_int8.zeros({kNumRows, kNumCols});

  // ============================================================
  // Mask tensor: single element (unused)
  // ============================================================
  Tensor mask = tf_int8.make({1}, {0});

  // ============================================================
  // Position tensor: single element (unused)
  // ============================================================
  Tensor pos = tf_int64.make({1}, {0});

  // ============================================================
  // Quantization parameters
  // ============================================================
  const double in_scale = 0.1;
  const int64_t in_zero_point = 0;
  const double out_scale = 1.0 / 127.0;
  const int64_t out_zero_point = 0;

  // ============================================================
  // Execute softmax
  // ============================================================
  quantized_softmax_per_tensor_out(
      input,
      mask,
      -1,
      0,
      pos,
      in_scale,
      in_zero_point,
      out_scale,
      out_zero_point,
      output);

  // ============================================================
  // Expected output tensor
  // Uniform input: all outputs should be approximately equal (1/16 per element)
  // Expected value: 127 / 16 ≈ 8
  // ============================================================
  // clang-format off
  Tensor expected = tf_int8.make(
      {kNumRows, kNumCols},
      {
          // Row 0: uniform output
          8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
          // Row 1
          8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
          // Row 2
          8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
          // Row 3
          8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8
      });
  // clang-format on

  EXPECT_TENSOR_EQ(output, expected);
}

// Test softmax with negative position value (all positions masked)
TEST_F(GenericQuantizedSoftmaxTest, SoftmaxAllMaskedInt8) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Long> tf_int64;

  // ============================================================
  // Tensor dimensions
  // ============================================================
  constexpr int kNumRows = 4;
  constexpr int kNumCols = 16;

  // ============================================================
  // Input tensor: various values
  // ============================================================
  // clang-format off
  Tensor input = tf_int8.make(
      {kNumRows, kNumCols},
      {
          // Row 0
          10, 20, 30, 40, 50, 60, 10, 20, 30, 40, 50, 60, 10, 20, 30, 40,
          // Row 1
          10, 20, 30, 40, 50, 60, 10, 20, 30, 40, 50, 60, 10, 20, 30, 40,
          // Row 2
          10, 20, 30, 40, 50, 60, 10, 20, 30, 40, 50, 60, 10, 20, 30, 40,
          // Row 3
          10, 20, 30, 40, 50, 60, 10, 20, 30, 40, 50, 60, 10, 20, 30, 40
      });
  // clang-format on

  // ============================================================
  // Output tensor: initialized to zeros
  // ============================================================
  Tensor output = tf_int8.zeros({kNumRows, kNumCols});

  // ============================================================
  // Mask tensor: single element (unused)
  // ============================================================
  Tensor mask = tf_int8.make({1}, {0});

  // ============================================================
  // Position tensor: -1 means all positions masked
  // ============================================================
  Tensor pos = tf_int64.make({1}, {-1});

  // ============================================================
  // Quantization parameters
  // ============================================================
  const double in_scale = 0.1;
  const int64_t in_zero_point = 0;
  const double out_scale = 1.0 / 127.0;
  const int64_t out_zero_point = 0;

  // ============================================================
  // Execute softmax
  // ============================================================
  quantized_softmax_per_tensor_out(
      input,
      mask,
      -1,
      1, // mask_type = 1
      pos,
      in_scale,
      in_zero_point,
      out_scale,
      out_zero_point,
      output);

  // ============================================================
  // Expected output tensor
  // All positions are masked, so all outputs should be zero
  // ============================================================
  // clang-format off
  Tensor expected = tf_int8.make(
      {kNumRows, kNumCols},
      {
          // Row 0: all masked
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          // Row 1
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          // Row 2
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          // Row 3
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
      });
  // clang-format on

  EXPECT_TENSOR_EQ(output, expected);
}

// Test softmax with large position value (no positions masked)
TEST_F(GenericQuantizedSoftmaxTest, SoftmaxNoneMaskedInt8) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Long> tf_int64;

  // ============================================================
  // Tensor dimensions
  // ============================================================
  constexpr int kNumRows = 4;
  constexpr int kNumCols = 16;

  // ============================================================
  // Input tensor: increasing pattern
  // ============================================================
  // clang-format off
  Tensor input = tf_int8.make(
      {kNumRows, kNumCols},
      {
          // Row 0
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
          // Row 1
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
          // Row 2
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
          // Row 3
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
      });
  // clang-format on

  // ============================================================
  // Output tensor: initialized to zeros
  // ============================================================
  Tensor output = tf_int8.zeros({kNumRows, kNumCols});

  // ============================================================
  // Mask tensor: single element (unused)
  // ============================================================
  Tensor mask = tf_int8.make({1}, {0});

  // ============================================================
  // Position tensor: 1000 > size, so no masking
  // ============================================================
  Tensor pos = tf_int64.make({1}, {1000});

  // ============================================================
  // Quantization parameters
  // ============================================================
  const double in_scale = 0.1;
  const int64_t in_zero_point = 0;
  const double out_scale = 1.0 / 127.0;
  const int64_t out_zero_point = 0;

  // ============================================================
  // Execute softmax
  // ============================================================
  quantized_softmax_per_tensor_out(
      input,
      mask,
      -1,
      1, // mask_type = 1
      pos,
      in_scale,
      in_zero_point,
      out_scale,
      out_zero_point,
      output);

  // ============================================================
  // Expected output tensor
  // No masking (mask_type=0), all positions attended
  // Same pattern as BasicSoftmaxInt8NoMask
  // ============================================================
  // clang-format off
  Tensor expected = tf_int8.make(
      {kNumRows, kNumCols},
      {
          // Row 0: increasing values, softmax distribution
          3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 11, 12, 14, 15,
          // Row 1
          3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 11, 12, 14, 15,
          // Row 2
          3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 11, 12, 14, 15,
          // Row 3
          3, 4, 4, 5, 5, 6, 6, 7, 8, 8, 9, 10, 11, 12, 14, 15
      });
  // clang-format on

  EXPECT_TENSOR_EQ(output, expected);
}

// Test softmax with single element per row
TEST_F(GenericQuantizedSoftmaxTest, SoftmaxSingleElementInt8) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Long> tf_int64;

  // ============================================================
  // Tensor dimensions: multiple rows, single column
  // ============================================================
  constexpr int kNumRows = 8;
  constexpr int kNumCols = 1;

  // ============================================================
  // Input tensor: various single values
  // ============================================================
  Tensor input =
      tf_int8.make({kNumRows, kNumCols}, {10, 20, 30, 40, 50, 60, 70, 80});

  // ============================================================
  // Output tensor: initialized to zeros
  // ============================================================
  Tensor output = tf_int8.zeros({kNumRows, kNumCols});

  // ============================================================
  // Mask tensor: single element (unused)
  // ============================================================
  Tensor mask = tf_int8.make({1}, {0});

  // ============================================================
  // Position tensor: single element (unused)
  // ============================================================
  Tensor pos = tf_int64.make({1}, {0});

  // ============================================================
  // Quantization parameters
  // ============================================================
  const double in_scale = 0.1;
  const int64_t in_zero_point = 0;
  const double out_scale = 1.0 / 127.0;
  const int64_t out_zero_point = 0;

  // ============================================================
  // Execute softmax
  // ============================================================
  quantized_softmax_per_tensor_out(
      input,
      mask,
      -1,
      0,
      pos,
      in_scale,
      in_zero_point,
      out_scale,
      out_zero_point,
      output);

  // Single element softmax should output 1.0 (quantized as 127 for all rows)
  // clang-format off
  Tensor expected = tf_int8.make(
      {kNumRows, kNumCols},
      {
          127,  // Row 0
          127,  // Row 1
          127,  // Row 2
          127,  // Row 3
          127,  // Row 4
          127,  // Row 5
          127,  // Row 6
          127,  // Row 7
      });
  // clang-format on
  EXPECT_TENSOR_EQ(output, expected);
}

// Test softmax with int16 position tensor for causal masking
TEST_F(GenericQuantizedSoftmaxTest, SoftmaxCausalMaskingInt16Pos) {
  TensorFactory<ScalarType::Short> tf_int16;

  // ============================================================
  // Tensor dimensions
  // ============================================================
  constexpr int kNumRows = 4;
  constexpr int kNumCols = 16;

  // ============================================================
  // Input tensor: increasing values
  // ============================================================
  // clang-format off
  Tensor input = tf_int16.make(
      {kNumRows, kNumCols},
      {
          // Row 0
          100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250,
          // Row 1
          100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250,
          // Row 2
          100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250,
          // Row 3
          100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250
      });
  // clang-format on

  // ============================================================
  // Output tensor: initialized to zeros
  // ============================================================
  Tensor output = tf_int16.zeros({kNumRows, kNumCols});

  // ============================================================
  // Mask tensor: single element (unused)
  // ============================================================
  Tensor mask = tf_int16.make({1}, {0});

  // ============================================================
  // Position tensor as int16: start from position 5
  // ============================================================
  Tensor pos = tf_int16.make({1}, {5});

  // ============================================================
  // Quantization parameters
  // ============================================================
  const double in_scale = 0.01;
  const int64_t in_zero_point = 0;
  const double out_scale = 1.0 / 32767.0;
  const int64_t out_zero_point = 0;

  // ============================================================
  // Execute softmax
  // ============================================================
  quantized_softmax_per_tensor_out(
      input,
      mask,
      -1,
      1, // mask_type = 1
      pos,
      in_scale,
      in_zero_point,
      out_scale,
      out_zero_point,
      output);

  // Compare with expected output tensor
  // Row 0 (pos=5): positions 0-5 attended (exponential growth), positions 6-15
  // masked (0) Row 1 (pos=5+1=6): positions 0-6 attended, positions 7-15 masked
  // Row 2 (pos=5+2=7): positions 0-7 attended, positions 8-15 masked
  // Row 3 (pos=5+3=8): positions 0-8 attended, positions 9-15 masked
  // clang-format off
  Tensor expected = tf_int16.make(
      {kNumRows, kNumCols},
      {
          // Row 0: positions 0-5 attended (exp growth), 6-15 masked
          4192, 4633, 5120, 5658, 6253, 6911, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          // Row 1: positions 0-6 attended, 7-15 masked
          3399, 3757, 4152, 4589, 5071, 5605, 6194, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          // Row 2: positions 0-7 attended, 8-15 masked
          2812, 3108, 3434, 3796, 4195, 4636, 5124, 5663, 0, 0, 0, 0, 0, 0, 0, 0,
          // Row 3: positions 0-8 attended, 9-15 masked
          2361, 2609, 2884, 3187, 3522, 3893, 4302, 4754, 5255, 0, 0, 0, 0, 0, 0, 0,
      });
  // clang-format on
  EXPECT_TENSOR_EQ(output, expected);
}

// Test numerical accuracy: verify softmax with known values
TEST_F(GenericQuantizedSoftmaxTest, SoftmaxNumericalAccuracyKnown) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Long> tf_int64;

  // ============================================================
  // Tensor dimensions: 2 rows x 4 cols
  // ============================================================
  constexpr int kNumRows = 2;
  constexpr int kNumCols = 4;

  // ============================================================
  // Input tensor: [0, 1, 2, 3] repeated
  // softmax([0, 1, 2, 3]) = [0.0321, 0.0871, 0.2369, 0.6439]
  // ============================================================
  Tensor input = tf_int8.make(
      {kNumRows, kNumCols},
      {// Row 0
       0,
       1,
       2,
       3,
       // Row 1
       0,
       1,
       2,
       3});

  // ============================================================
  // Output tensor: initialized to zeros
  // ============================================================
  Tensor output = tf_int8.zeros({kNumRows, kNumCols});

  // ============================================================
  // Mask tensor: single element (unused)
  // ============================================================
  Tensor mask = tf_int8.make({1}, {0});

  // ============================================================
  // Position tensor: single element (unused)
  // ============================================================
  Tensor pos = tf_int64.make({1}, {0});

  // ============================================================
  // Quantization parameters
  // With out_scale = 0.01, expected quantized values: [3, 9, 24, 64]
  // ============================================================
  const double in_scale = 1.0;
  const int64_t in_zero_point = 0;
  const double out_scale = 0.01;
  const int64_t out_zero_point = 0;

  // ============================================================
  // Execute softmax
  // ============================================================
  quantized_softmax_per_tensor_out(
      input,
      mask,
      -1,
      0,
      pos,
      in_scale,
      in_zero_point,
      out_scale,
      out_zero_point,
      output);

  // Compare with expected output tensor
  // Both rows have identical input [0, 1, 2, 3], so both have the same
  // softmax distribution: approx [0.032, 0.087, 0.237, 0.644]
  // Quantized with scale=0.01, zero_point=0: [3, 9, 24, 64]
  // clang-format off
  Tensor expected = tf_int8.make(
      {kNumRows, kNumCols},
      {
          // Row 0: softmax([0,1,2,3]) quantized
          3, 9, 24, 64,
          // Row 1: identical distribution
          3, 9, 24, 64,
      });
  // clang-format on
  EXPECT_TENSOR_EQ(output, expected);
}

// Test causal masking with larger sequence for attention-like patterns
TEST_F(GenericQuantizedSoftmaxTest, SoftmaxCausalMaskingLargeSequence) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Long> tf_int64;

  // ============================================================
  // Tensor dimensions: simulating multi-head attention
  // ============================================================
  constexpr int kNumHeads = 4;
  constexpr int kSeqLen = 32;

  // ============================================================
  // Input tensor: attention scores (128 elements)
  // ============================================================
  // clang-format off
  Tensor input = tf_int8.make(
      {kNumHeads, kSeqLen},
      {
          // Head 0
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
          // Head 1
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
          // Head 2
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
          // Head 3
          10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41
      });
  // clang-format on

  // ============================================================
  // Output tensor: initialized to zeros
  // ============================================================
  Tensor output = tf_int8.zeros({kNumHeads, kSeqLen});

  // ============================================================
  // Mask tensor: single element (unused)
  // ============================================================
  Tensor mask = tf_int8.make({1}, {0});

  // ============================================================
  // Position tensor: start with position 7
  // ============================================================
  Tensor pos = tf_int64.make({1}, {7});

  // ============================================================
  // Quantization parameters
  // ============================================================
  const double in_scale = 0.1;
  const int64_t in_zero_point = 0;
  const double out_scale = 1.0 / 127.0;
  const int64_t out_zero_point = 0;

  // ============================================================
  // Execute softmax
  // ============================================================
  quantized_softmax_per_tensor_out(
      input,
      mask,
      -1,
      1, // Causal masking
      pos,
      in_scale,
      in_zero_point,
      out_scale,
      out_zero_point,
      output);

  // Compare with expected output tensor
  // Head 0 (pos=7): positions 0-7 attended, positions 8-31 masked
  // Head 1 (pos=8): positions 0-8 attended, positions 9-31 masked
  // Head 2 (pos=9): positions 0-9 attended, positions 10-31 masked
  // Head 3 (pos=10): positions 0-10 attended, positions 11-31 masked
  // clang-format off
  Tensor expected = tf_int8.make(
      {kNumHeads, kSeqLen},
      {
          // Head 0: positions 0-7 attended, 8-31 masked
          11, 12, 13, 15, 16, 18, 20, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          // Head 1: positions 0-8 attended, 9-31 masked
          9, 10, 11, 12, 14, 15, 17, 18, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          // Head 2: positions 0-9 attended, 10-31 masked
          8, 9, 9, 10, 12, 13, 14, 16, 17, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          // Head 3: positions 0-10 attended, 11-31 masked
          7, 7, 8, 9, 10, 11, 12, 13, 15, 16, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      });
  // clang-format on
  EXPECT_TENSOR_EQ(output, expected);
}

// ============================================================
// Test softmax with basePosValue = 0 (only first position attended)
// ============================================================
TEST_F(GenericQuantizedSoftmaxTest, SoftmaxFirstPositionOnlyInt8) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Long> tf_int64;

  // ============================================================
  // Tensor dimensions
  // ============================================================
  constexpr int kNumRows = 4;
  constexpr int kNumCols = 8;

  // ============================================================
  // Input tensor: 4 rows x 8 cols
  // ============================================================
  // clang-format off
  Tensor input = tf_int8.make(
      {kNumRows, kNumCols},
      {
          // Row 0
          50, 40, 30, 20, 10, 10, 10, 10,
          // Row 1
          50, 40, 30, 20, 10, 10, 10, 10,
          // Row 2
          50, 40, 30, 20, 10, 10, 10, 10,
          // Row 3
          50, 40, 30, 20, 10, 10, 10, 10
      });
  // clang-format on

  // ============================================================
  // Output tensor: initialized to zeros
  // ============================================================
  Tensor output = tf_int8.zeros({kNumRows, kNumCols});

  // ============================================================
  // Mask tensor: unused
  // ============================================================
  Tensor mask = tf_int8.make({1}, {0});

  // ============================================================
  // Position tensor: 0 means only first position attended
  // Row 0: pos=0 (only col 0 attended)
  // Row 1: pos=1 (cols 0-1 attended)
  // Row 2: pos=2 (cols 0-2 attended)
  // Row 3: pos=3 (cols 0-3 attended)
  // ============================================================
  Tensor pos = tf_int64.make({1}, {0});

  // ============================================================
  // Quantization parameters
  // ============================================================
  const double in_scale = 0.1;
  const int64_t in_zero_point = 0;
  const double out_scale = 1.0 / 127.0;
  const int64_t out_zero_point = 0;

  // ============================================================
  // Expected output tensor
  // Row 0: pos=0, only col 0 attended -> softmax([5.0]) = [1.0] -> [127]
  // Row 1: pos=1, cols 0-1 attended -> softmax([5.0, 4.0])
  //        = exp([5,4]) / sum = [148.4, 54.6] / 203 = [0.73, 0.27]
  //        Quantized at scale=1/127: [93, 34]
  // Row 2: pos=2, cols 0-2 attended -> softmax([5.0, 4.0, 3.0])
  //        = exp([5,4,3]) / sum = [148.4, 54.6, 20.1] / 223.1 = [0.665, 0.245,
  //        0.09] Quantized at scale=1/127: [84, 31, 11]
  // Row 3: pos=3, cols 0-3 attended -> softmax([5.0, 4.0, 3.0, 2.0])
  //        = exp([5,4,3,2]) / sum = [148.4, 54.6, 20.1, 7.4] / 230.5 = [0.644,
  //        0.237, 0.087, 0.032] Quantized at scale=1/127: [82, 30, 11, 4]
  // Masked positions get out_zero_point = 0
  // ============================================================
  // clang-format off
  Tensor expected = tf_int8.make(
      {kNumRows, kNumCols},
      {
          // Row 0: only col 0 attended
          127, 0, 0, 0, 0, 0, 0, 0,
          // Row 1: cols 0-1 attended
          93, 34, 0, 0, 0, 0, 0, 0,
          // Row 2: cols 0-2 attended
          84, 31, 11, 0, 0, 0, 0, 0,
          // Row 3: cols 0-3 attended
          82, 30, 11, 4, 0, 0, 0, 0
      });
  // clang-format on

  // ============================================================
  // Execute softmax
  // ============================================================
  quantized_softmax_per_tensor_out(
      input,
      mask,
      -1,
      1, // mask_type = 1
      pos,
      in_scale,
      in_zero_point,
      out_scale,
      out_zero_point,
      output);

  EXPECT_TENSOR_EQ(output, expected);
}

// ============================================================
// Test softmax with single row
// ============================================================
TEST_F(GenericQuantizedSoftmaxTest, SoftmaxSingleRowInt8) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Long> tf_int64;

  // ============================================================
  // Tensor dimensions
  // ============================================================
  constexpr int kNumCols = 8;

  // ============================================================
  // Input tensor: 1 row x 8 cols
  // ============================================================
  // clang-format off
  Tensor input = tf_int8.make(
      {1, kNumCols},
      {
          100, 90, 80, 70, 60, 50, 40, 30
      });
  // clang-format on

  // ============================================================
  // Output tensor: initialized to zeros
  // ============================================================
  Tensor output = tf_int8.zeros({1, kNumCols});

  // ============================================================
  // Mask tensor: unused
  // ============================================================
  Tensor mask = tf_int8.make({1}, {0});

  // ============================================================
  // Position tensor: 3 means positions 0-3 are attended
  // ============================================================
  Tensor pos = tf_int64.make({1}, {3});

  // ============================================================
  // Quantization parameters
  // ============================================================
  const double in_scale = 0.1;
  const int64_t in_zero_point = 0;
  const double out_scale = 1.0 / 127.0;
  const int64_t out_zero_point = 0;

  // ============================================================
  // Expected output tensor
  // pos=3: cols 0-3 attended
  // Input dequantized values: [10.0, 9.0, 8.0, 7.0, ...]
  // softmax([10.0, 9.0, 8.0, 7.0]) = [0.643, 0.236, 0.087, 0.032]
  // Quantized at out_scale = 1/127: [82, 30, 11, 4]
  // Masked positions get out_zero_point = 0
  // ============================================================
  // clang-format off
  Tensor expected = tf_int8.make(
      {1, kNumCols},
      {
          82, 30, 11, 4, 0, 0, 0, 0
      });
  // clang-format on

  // ============================================================
  // Execute softmax
  // ============================================================
  quantized_softmax_per_tensor_out(
      input,
      mask,
      -1,
      1, // mask_type = 1
      pos,
      in_scale,
      in_zero_point,
      out_scale,
      out_zero_point,
      output);

  EXPECT_TENSOR_EQ(output, expected);
}

// ============================================================
// Test softmax with Int16 position tensor
// ============================================================
TEST_F(GenericQuantizedSoftmaxTest, SoftmaxWithInt16PositionTensor) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Short> tf_int16;

  // ============================================================
  // Tensor dimensions
  // ============================================================
  constexpr int kNumRows = 2;
  constexpr int kNumCols = 8;

  // ============================================================
  // Input tensor: 2 rows x 8 cols
  // ============================================================
  // clang-format off
  Tensor input = tf_int8.make(
      {kNumRows, kNumCols},
      {
          // Row 0
          10, 20, 30, 40, 50, 60, 70, 80,
          // Row 1
          10, 20, 30, 40, 50, 60, 70, 80
      });
  // clang-format on

  // ============================================================
  // Output tensor: initialized to zeros
  // ============================================================
  Tensor output = tf_int8.zeros({kNumRows, kNumCols});

  // ============================================================
  // Mask tensor: unused
  // ============================================================
  Tensor mask = tf_int8.make({1}, {0});

  // ============================================================
  // Position tensor: Int16 type with value 2
  // Row 0: pos=2 (cols 0-2 attended)
  // Row 1: pos=3 (cols 0-3 attended)
  // ============================================================
  Tensor pos = tf_int16.make({1}, {2});

  // ============================================================
  // Quantization parameters
  // ============================================================
  const double in_scale = 0.1;
  const int64_t in_zero_point = 0;
  const double out_scale = 1.0 / 127.0;
  const int64_t out_zero_point = 0;

  // ============================================================
  // Expected output tensor
  // Row 0: pos=2, cols 0-2 attended
  // Input dequantized values: [1.0, 2.0, 3.0, ...]
  // softmax([1.0, 2.0, 3.0]) = exp([1,2,3]) / sum
  // Quantized at out_scale = 1/127 (values verified from implementation)
  // Row 1: pos=3, cols 0-3 attended
  // softmax([1.0, 2.0, 3.0, 4.0]) = exp([1,2,3,4]) / sum
  // Masked positions get out_zero_point = 0
  // ============================================================
  // clang-format off
  Tensor expected = tf_int8.make(
      {kNumRows, kNumCols},
      {
          // Row 0: cols 0-2 attended
          11, 31, 84, 0, 0, 0, 0, 0,
          // Row 1: cols 0-3 attended
          4, 11, 30, 82, 0, 0, 0, 0
      });
  // clang-format on

  // ============================================================
  // Execute softmax
  // ============================================================
  quantized_softmax_per_tensor_out(
      input,
      mask,
      -1,
      1, // mask_type = 1
      pos,
      in_scale,
      in_zero_point,
      out_scale,
      out_zero_point,
      output);

  EXPECT_TENSOR_EQ(output, expected);
}

// ============================================================
// Test softmax with non-zero output zero_point
// ============================================================
TEST_F(GenericQuantizedSoftmaxTest, SoftmaxWithNonZeroOutputZeroPoint) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Long> tf_int64;

  // ============================================================
  // Tensor dimensions
  // ============================================================
  constexpr int kNumRows = 2;
  constexpr int kNumCols = 8;

  // ============================================================
  // Input tensor
  // ============================================================
  // clang-format off
  Tensor input = tf_int8.make(
      {kNumRows, kNumCols},
      {
          // Row 0
          10, 20, 30, 40, 50, 60, 70, 80,
          // Row 1
          10, 20, 30, 40, 50, 60, 70, 80
      });
  // clang-format on

  // ============================================================
  // Output tensor: initialized to zeros
  // ============================================================
  Tensor output = tf_int8.zeros({kNumRows, kNumCols});

  // ============================================================
  // Mask and position tensors
  // ============================================================
  Tensor mask = tf_int8.make({1}, {0});
  Tensor pos = tf_int64.make({1}, {-1}); // All masked

  // ============================================================
  // Quantization parameters with non-zero output zero_point
  // ============================================================
  const double in_scale = 0.1;
  const int64_t in_zero_point = 0;
  const double out_scale = 1.0 / 127.0;
  const int64_t out_zero_point = 64; // Non-zero zero_point

  // ============================================================
  // Expected output tensor
  // All positions masked -> all outputs should be out_zero_point = 64
  // ============================================================
  // clang-format off
  Tensor expected = tf_int8.make(
      {kNumRows, kNumCols},
      {
          // Row 0: all masked
          64, 64, 64, 64, 64, 64, 64, 64,
          // Row 1: all masked
          64, 64, 64, 64, 64, 64, 64, 64
      });
  // clang-format on

  // ============================================================
  // Execute softmax
  // ============================================================
  quantized_softmax_per_tensor_out(
      input,
      mask,
      -1,
      1, // mask_type = 1
      pos,
      in_scale,
      in_zero_point,
      out_scale,
      out_zero_point,
      output);

  EXPECT_TENSOR_EQ(output, expected);
}

// ============================================================
// Test full causal masking pattern (simulates transformer attention)
// ============================================================
TEST_F(GenericQuantizedSoftmaxTest, CausalMaskingSimulationInt8) {
  TensorFactory<ScalarType::Char> tf_int8;
  TensorFactory<ScalarType::Long> tf_int64;

  // ============================================================
  // Tensor dimensions - 8x8 simulates 8 token positions
  // ============================================================
  constexpr int kNumRows = 8;
  constexpr int kNumCols = 8;

  // ============================================================
  // Input tensor: uniform values (no preference for any position)
  // All values are the same to make expected output predictable
  // ============================================================
  // clang-format off
  Tensor input = tf_int8.make(
      {kNumRows, kNumCols},
      {
          50, 50, 50, 50, 50, 50, 50, 50,
          50, 50, 50, 50, 50, 50, 50, 50,
          50, 50, 50, 50, 50, 50, 50, 50,
          50, 50, 50, 50, 50, 50, 50, 50,
          50, 50, 50, 50, 50, 50, 50, 50,
          50, 50, 50, 50, 50, 50, 50, 50,
          50, 50, 50, 50, 50, 50, 50, 50,
          50, 50, 50, 50, 50, 50, 50, 50
      });
  // clang-format on

  // ============================================================
  // Output tensor: initialized to zeros
  // ============================================================
  Tensor output = tf_int8.zeros({kNumRows, kNumCols});

  // ============================================================
  // Mask tensor: unused
  // ============================================================
  Tensor mask = tf_int8.make({1}, {0});

  // ============================================================
  // Position tensor: 0 means first token
  // Row 0: pos=0, cols 0 attended
  // Row 1: pos=1, cols 0-1 attended
  // ...
  // Row 7: pos=7, cols 0-7 attended
  // ============================================================
  Tensor pos = tf_int64.make({1}, {0});

  // ============================================================
  // Quantization parameters
  // ============================================================
  const double in_scale = 0.1;
  const int64_t in_zero_point = 0;
  const double out_scale = 1.0 / 127.0;
  const int64_t out_zero_point = 0;

  // ============================================================
  // Expected output tensor
  // With uniform inputs, softmax distributes probability evenly
  // among attended positions:
  // Row 0: 1 attended -> [127, 0, 0, 0, 0, 0, 0, 0]
  // Row 1: 2 attended -> [64, 64, 0, 0, 0, 0, 0, 0]
  // Row 2: 3 attended -> [42, 42, 42, 0, 0, 0, 0, 0]
  // Row 3: 4 attended -> [32, 32, 32, 32, 0, 0, 0, 0]
  // Row 4: 5 attended -> [25, 25, 25, 25, 25, 0, 0, 0]
  // Row 5: 6 attended -> [21, 21, 21, 21, 21, 21, 0, 0]
  // Row 6: 7 attended -> [18, 18, 18, 18, 18, 18, 18, 0]
  // Row 7: 8 attended -> [16, 16, 16, 16, 16, 16, 16, 16]
  // Note: These are exact quantized values (127/n rounded)
  // ============================================================
  // clang-format off
  Tensor expected = tf_int8.make(
      {kNumRows, kNumCols},
      {
          // Row 0: 1 attended (127/1 = 127)
          127, 0, 0, 0, 0, 0, 0, 0,
          // Row 1: 2 attended (127/2 ≈ 63-64)
          64, 64, 0, 0, 0, 0, 0, 0,
          // Row 2: 3 attended (127/3 ≈ 42)
          42, 42, 42, 0, 0, 0, 0, 0,
          // Row 3: 4 attended (127/4 ≈ 32)
          32, 32, 32, 32, 0, 0, 0, 0,
          // Row 4: 5 attended (127/5 ≈ 25)
          25, 25, 25, 25, 25, 0, 0, 0,
          // Row 5: 6 attended (127/6 ≈ 21)
          21, 21, 21, 21, 21, 21, 0, 0,
          // Row 6: 7 attended (127/7 ≈ 18)
          18, 18, 18, 18, 18, 18, 18, 0,
          // Row 7: 8 attended (127/8 ≈ 16)
          16, 16, 16, 16, 16, 16, 16, 16
      });
  // clang-format on

  // ============================================================
  // Execute softmax
  // ============================================================
  quantized_softmax_per_tensor_out(
      input,
      mask,
      -1,
      1, // mask_type = 1
      pos,
      in_scale,
      in_zero_point,
      out_scale,
      out_zero_point,
      output);

  EXPECT_TENSOR_EQ(output, expected);
}

} // namespace
} // namespace native
} // namespace generic
} // namespace impl
