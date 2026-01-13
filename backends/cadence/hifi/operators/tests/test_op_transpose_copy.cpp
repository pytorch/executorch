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
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::aten::TensorImpl;
using ::executorch::runtime::Error;
using ::executorch::runtime::KernelRuntimeContext;
using ::executorch::runtime::runtime_init;
using ::executorch::runtime::testing::TensorFactory;

class HiFiTransposeCopyTest : public OperatorTest {
 public:
 protected:
  Tensor& transpose_copy_int_out(
      const Tensor& in,
      int64_t dim0,
      int64_t dim1,
      Tensor& out) {
    return ::impl::HiFi::native::transpose_copy_int_out(
        context_, in, dim0, dim1, out);
  }

  // Helper to verify transpose correctness by checking specific elements
  template <typename T>
  bool verifyTranspose2D(
      const Tensor& in,
      const Tensor& out,
      int64_t dim0,
      int64_t dim1) {
    // For 2D tensors, verify that out[j][i] == in[i][j]
    const T* in_data = in.const_data_ptr<T>();
    const T* out_data = out.const_data_ptr<T>();

    int rows = in.size(0);
    int cols = in.size(1);

    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        // in[i][j] should equal out[j][i]
        if (in_data[i * cols + j] != out_data[j * rows + i]) {
          return false;
        }
      }
    }
    return true;
  }
};

// Test basic 2D float transpose (matrix transpose)
// Verifies that the optimized xa_nn_transpose_32_32 path works correctly
TEST_F(HiFiTransposeCopyTest, Basic2DFloatTranspose) {
  TensorFactory<ScalarType::Float> tf;

  // Input: (2, 3) matrix with sequential values
  const std::vector<int32_t> input_sizes{2, 3};
  const std::vector<int32_t> output_sizes{3, 2};

  // Input matrix:
  // [[1, 2, 3],
  //  [4, 5, 6]]
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  // Expected transposed:
  // [[1, 4],
  //  [2, 5],
  //  [3, 6]]
  std::vector<float> expected_data = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};

  Tensor input = tf.make(input_sizes, input_data);
  Tensor out = tf.zeros(output_sizes);
  Tensor expected = tf.make(output_sizes, expected_data);

  transpose_copy_int_out(input, 0, 1, out);

  EXPECT_TENSOR_CLOSE(out, expected);
  EXPECT_TRUE(verifyTranspose2D<float>(input, out, 0, 1));
}

// Test 2D int8 transpose - uses optimized xa_nn_transpose_8_8 path
TEST_F(HiFiTransposeCopyTest, Basic2DInt8Transpose) {
  TensorFactory<ScalarType::Char> tf;

  const std::vector<int32_t> input_sizes{2, 3};
  const std::vector<int32_t> output_sizes{3, 2};

  std::vector<int8_t> input_data = {1, 2, 3, 4, 5, 6};
  std::vector<int8_t> expected_data = {1, 4, 2, 5, 3, 6};

  Tensor input = tf.make(input_sizes, input_data);
  Tensor out = tf.zeros(output_sizes);
  Tensor expected = tf.make(output_sizes, expected_data);

  transpose_copy_int_out(input, 0, 1, out);

  EXPECT_TENSOR_CLOSE(out, expected);
}

// Test 2D uint8 transpose - uses optimized xa_nn_transpose_8_8 path
TEST_F(HiFiTransposeCopyTest, Basic2DUInt8Transpose) {
  TensorFactory<ScalarType::Byte> tf;

  const std::vector<int32_t> input_sizes{2, 3};
  const std::vector<int32_t> output_sizes{3, 2};

  std::vector<uint8_t> input_data = {10, 20, 30, 40, 50, 60};
  std::vector<uint8_t> expected_data = {10, 40, 20, 50, 30, 60};

  Tensor input = tf.make(input_sizes, input_data);
  Tensor out = tf.zeros(output_sizes);
  Tensor expected = tf.make(output_sizes, expected_data);

  transpose_copy_int_out(input, 0, 1, out);

  EXPECT_TENSOR_CLOSE(out, expected);
}

// Test 3D transpose swapping first two dimensions
TEST_F(HiFiTransposeCopyTest, Transpose3DDim0Dim1) {
  TensorFactory<ScalarType::Float> tf;

  // Input: (2, 3, 4) -> Output: (3, 2, 4)
  const std::vector<int32_t> input_sizes{2, 3, 4};
  const std::vector<int32_t> output_sizes{3, 2, 4};

  Tensor input = tf.ones(input_sizes);
  Tensor out = tf.zeros(output_sizes);

  transpose_copy_int_out(input, 0, 1, out);

  // Verify output shape
  EXPECT_EQ(out.size(0), 3);
  EXPECT_EQ(out.size(1), 2);
  EXPECT_EQ(out.size(2), 4);

  // Verify all values are preserved (ones input -> ones output)
  const float* out_data = out.const_data_ptr<float>();
  for (int i = 0; i < out.numel(); ++i) {
    EXPECT_EQ(out_data[i], 1.0f);
  }
}

// Test 3D transpose swapping last two dimensions
TEST_F(HiFiTransposeCopyTest, Transpose3DDim1Dim2) {
  TensorFactory<ScalarType::Float> tf;

  // Input: (2, 3, 4) -> Output: (2, 4, 3)
  const std::vector<int32_t> input_sizes{2, 3, 4};
  const std::vector<int32_t> output_sizes{2, 4, 3};

  Tensor input = tf.ones(input_sizes);
  Tensor out = tf.zeros(output_sizes);

  transpose_copy_int_out(input, 1, 2, out);

  EXPECT_EQ(out.size(0), 2);
  EXPECT_EQ(out.size(1), 4);
  EXPECT_EQ(out.size(2), 3);

  // Verify all values are preserved
  const float* out_data = out.const_data_ptr<float>();
  for (int i = 0; i < out.numel(); ++i) {
    EXPECT_EQ(out_data[i], 1.0f);
  }
}

// Test 3D transpose swapping first and last dimensions
TEST_F(HiFiTransposeCopyTest, Transpose3DDim0Dim2) {
  TensorFactory<ScalarType::Float> tf;

  // Input: (2, 3, 4) -> Output: (4, 3, 2)
  const std::vector<int32_t> input_sizes{2, 3, 4};
  const std::vector<int32_t> output_sizes{4, 3, 2};

  Tensor input = tf.ones(input_sizes);
  Tensor out = tf.zeros(output_sizes);

  transpose_copy_int_out(input, 0, 2, out);

  EXPECT_EQ(out.size(0), 4);
  EXPECT_EQ(out.size(1), 3);
  EXPECT_EQ(out.size(2), 2);
}

// Test 4D transpose (common in batch normalization and conv operations)
// Use sizes aligned to 4 bytes for Float type on Xtensa
TEST_F(HiFiTransposeCopyTest, Transpose4DNCHW) {
  TensorFactory<ScalarType::Float> tf;

  // Input: (2, 4, 4, 4) NCHW-like -> swap C and H -> (2, 4, 4, 4)
  // Using aligned dimensions to avoid alignment issues on Xtensa
  const std::vector<int32_t> input_sizes{2, 4, 4, 4};
  const std::vector<int32_t> output_sizes{2, 4, 4, 4};

  Tensor input = tf.ones(input_sizes);
  Tensor out = tf.zeros(output_sizes);

  transpose_copy_int_out(input, 1, 2, out);

  EXPECT_EQ(out.size(0), 2);
  EXPECT_EQ(out.size(1), 4);
  EXPECT_EQ(out.size(2), 4);
  EXPECT_EQ(out.size(3), 4);
}

// Test with negative dimension indices
TEST_F(HiFiTransposeCopyTest, NegativeDimensionIndices) {
  TensorFactory<ScalarType::Float> tf;

  // Input: (2, 3, 4) with dims -2 and -1 should be equivalent to dims 1 and 2
  const std::vector<int32_t> input_sizes{2, 3, 4};
  const std::vector<int32_t> output_sizes{2, 4, 3};

  std::vector<float> input_data(24);
  for (int i = 0; i < 24; ++i) {
    input_data[i] = static_cast<float>(i);
  }

  Tensor input = tf.make(input_sizes, input_data);
  Tensor out_negative = tf.zeros(output_sizes);
  Tensor out_positive = tf.zeros(output_sizes);

  // Use negative indices
  transpose_copy_int_out(input, -2, -1, out_negative);
  // Use positive indices (should give same result)
  transpose_copy_int_out(input, 1, 2, out_positive);

  EXPECT_TENSOR_CLOSE(out_negative, out_positive);
}

// Test square matrix transpose (special case)
TEST_F(HiFiTransposeCopyTest, SquareMatrixTranspose) {
  TensorFactory<ScalarType::Float> tf;

  // Input: (3, 3) square matrix
  const std::vector<int32_t> sizes{3, 3};

  // Input matrix:
  // [[1, 2, 3],
  //  [4, 5, 6],
  //  [7, 8, 9]]
  std::vector<float> input_data = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f};
  // Expected transposed:
  // [[1, 4, 7],
  //  [2, 5, 8],
  //  [3, 6, 9]]
  std::vector<float> expected_data = {
      1.0f, 4.0f, 7.0f,
      2.0f, 5.0f, 8.0f,
      3.0f, 6.0f, 9.0f};

  Tensor input = tf.make(sizes, input_data);
  Tensor out = tf.zeros(sizes);
  Tensor expected = tf.make(sizes, expected_data);

  transpose_copy_int_out(input, 0, 1, out);

  EXPECT_TENSOR_CLOSE(out, expected);
}

// Test large tensor transpose (stress test)
TEST_F(HiFiTransposeCopyTest, LargeTensorTranspose) {
  TensorFactory<ScalarType::Float> tf;

  // Input: (32, 64) - larger tensor to test performance path
  const std::vector<int32_t> input_sizes{32, 64};
  const std::vector<int32_t> output_sizes{64, 32};

  Tensor input = tf.ones(input_sizes);
  Tensor out = tf.zeros(output_sizes);

  transpose_copy_int_out(input, 0, 1, out);

  // Verify shape
  EXPECT_EQ(out.size(0), 64);
  EXPECT_EQ(out.size(1), 32);

  // Verify all values are preserved
  const float* out_data = out.const_data_ptr<float>();
  for (int i = 0; i < out.numel(); ++i) {
    EXPECT_EQ(out_data[i], 1.0f);
  }
}

// Test int8 with actual quantized-like values
TEST_F(HiFiTransposeCopyTest, Int8QuantizedValues) {
  TensorFactory<ScalarType::Char> tf;

  const std::vector<int32_t> input_sizes{4, 4};
  const std::vector<int32_t> output_sizes{4, 4};

  // Simulate quantized values (typical range for int8 quantization)
  std::vector<int8_t> input_data = {
      -128, -64, 0, 64,
      127, -100, 50, -25,
      10, -10, 20, -20,
      30, -30, 40, -40};
  std::vector<int8_t> expected_data = {
      -128, 127, 10, 30,
      -64, -100, -10, -30,
      0, 50, 20, 40,
      64, -25, -20, -40};

  Tensor input = tf.make(input_sizes, input_data);
  Tensor out = tf.zeros(output_sizes);
  Tensor expected = tf.make(output_sizes, expected_data);

  transpose_copy_int_out(input, 0, 1, out);

  EXPECT_TENSOR_CLOSE(out, expected);
}

// Test 4D transpose shape (2, 3, 5, 7) with dim0=0, dim1=1
// Reproduces the failing test case test_aten_transpose_copy_int_5
TEST_F(HiFiTransposeCopyTest, Transpose4D_2_3_5_7_Dim01) {
  TensorFactory<ScalarType::Float> tf;

  // Input: (2, 3, 5, 7) -> Output: (3, 2, 5, 7) after swapping dims 0 and 1
  const std::vector<int32_t> input_sizes{2, 3, 5, 7};
  const std::vector<int32_t> output_sizes{3, 2, 5, 7};

  Tensor input = tf.ones(input_sizes);
  Tensor out = tf.zeros(output_sizes);

  transpose_copy_int_out(input, 0, 1, out);

  // Verify output shape
  EXPECT_EQ(out.size(0), 3);
  EXPECT_EQ(out.size(1), 2);
  EXPECT_EQ(out.size(2), 5);
  EXPECT_EQ(out.size(3), 7);

  // Verify all values are preserved (ones input -> ones output)
  const float* out_data = out.const_data_ptr<float>();
  for (int i = 0; i < out.numel(); ++i) {
    EXPECT_EQ(out_data[i], 1.0f);
  }
}

// Test preserving values through transpose (round-trip verification)
TEST_F(HiFiTransposeCopyTest, TransposePreservesValues) {
  TensorFactory<ScalarType::Float> tf;

  const std::vector<int32_t> input_sizes{3, 5};
  const std::vector<int32_t> transposed_sizes{5, 3};

  // Create input with distinct values
  std::vector<float> input_data(15);
  for (int i = 0; i < 15; ++i) {
    input_data[i] = static_cast<float>(i * 2 + 1);  // 1, 3, 5, 7, ...
  }

  Tensor input = tf.make(input_sizes, input_data);
  Tensor transposed = tf.zeros(transposed_sizes);
  Tensor restored = tf.zeros(input_sizes);

  // First transpose
  transpose_copy_int_out(input, 0, 1, transposed);
  // Second transpose (should restore original)
  transpose_copy_int_out(transposed, 0, 1, restored);

  // Verify round-trip preserves all values
  EXPECT_TENSOR_CLOSE(input, restored);
}

} // namespace
} // namespace native
} // namespace HiFi
} // namespace impl
