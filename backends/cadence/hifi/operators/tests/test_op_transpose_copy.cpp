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

  template <typename T>
  bool checkTransposed(
      const Tensor& in,
      const Tensor& out,
      int64_t dim0,
      int64_t dim1) {
    if (in.dim() != out.dim()) {
      return false;
    }
    if (in.size(dim0) != out.size(dim1) || in.size(dim1) != out.size(dim0)) {
      return false;
    }
    return true;
  }
};

// Test basic 2D transpose (matrix transpose)
TEST_F(HiFiTransposeCopyTest, Basic2DTranspose) {
  TensorFactory<ScalarType::Float> tf;

  // Input: (2, 3) matrix
  const std::vector<int32_t> input_sizes{2, 3};
  const std::vector<int32_t> output_sizes{3, 2};

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<float> expected_data = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};

  Tensor input = tf.make(input_sizes, input_data);
  Tensor out = tf.zeros(output_sizes);
  Tensor expected = tf.make(output_sizes, expected_data);

  transpose_copy_int_out(input, 0, 1, out);

  EXPECT_TRUE(checkTransposed<float>(input, out, 0, 1));
  EXPECT_TENSOR_CLOSE(out, expected);
}

// Test 3D transpose swapping first two dimensions
TEST_F(HiFiTransposeCopyTest, Transpose3DDim0Dim1) {
  TensorFactory<ScalarType::Float> tf;

  // Input: (2, 3, 4)
  const std::vector<int32_t> input_sizes{2, 3, 4};
  const std::vector<int32_t> output_sizes{3, 2, 4};

  Tensor input = tf.ones(input_sizes);
  Tensor out = tf.zeros(output_sizes);

  transpose_copy_int_out(input, 0, 1, out);

  EXPECT_TRUE(checkTransposed<float>(input, out, 0, 1));
  EXPECT_EQ(out.size(0), 3);
  EXPECT_EQ(out.size(1), 2);
  EXPECT_EQ(out.size(2), 4);
}

// Test 3D transpose swapping last two dimensions
TEST_F(HiFiTransposeCopyTest, Transpose3DDim1Dim2) {
  TensorFactory<ScalarType::Float> tf;

  // Input: (2, 3, 4)
  const std::vector<int32_t> input_sizes{2, 3, 4};
  const std::vector<int32_t> output_sizes{2, 4, 3};

  Tensor input = tf.ones(input_sizes);
  Tensor out = tf.zeros(output_sizes);

  transpose_copy_int_out(input, 1, 2, out);

  EXPECT_TRUE(checkTransposed<float>(input, out, 1, 2));
  EXPECT_EQ(out.size(0), 2);
  EXPECT_EQ(out.size(1), 4);
  EXPECT_EQ(out.size(2), 3);
}

// Test 4D transpose (common in batch normalization)
TEST_F(HiFiTransposeCopyTest, Transpose4D) {
  TensorFactory<ScalarType::Float> tf;

  // Input: (2, 3, 4, 5) - NCHW-like
  const std::vector<int32_t> input_sizes{2, 3, 4, 5};
  const std::vector<int32_t> output_sizes{2, 4, 3, 5};

  Tensor input = tf.ones(input_sizes);
  Tensor out = tf.zeros(output_sizes);

  transpose_copy_int_out(input, 1, 2, out);

  EXPECT_TRUE(checkTransposed<float>(input, out, 1, 2));
  EXPECT_EQ(out.size(0), 2);
  EXPECT_EQ(out.size(1), 4);
  EXPECT_EQ(out.size(2), 3);
  EXPECT_EQ(out.size(3), 5);
}

// Test with negative dimension indices
TEST_F(HiFiTransposeCopyTest, NegativeDimensions) {
  TensorFactory<ScalarType::Float> tf;

  // Input: (2, 3, 4)
  const std::vector<int32_t> input_sizes{2, 3, 4};
  const std::vector<int32_t> output_sizes{2, 4, 3};

  Tensor input = tf.ones(input_sizes);
  Tensor out = tf.zeros(output_sizes);

  transpose_copy_int_out(input, -2, -1, out);

  EXPECT_EQ(out.size(0), 2);
  EXPECT_EQ(out.size(1), 4);
  EXPECT_EQ(out.size(2), 3);
}

// Test int8 (Char) data type - uses optimized xa_nn_transpose_8_8
TEST_F(HiFiTransposeCopyTest, Int8Transpose) {
  TensorFactory<ScalarType::Char> tf;

  // Input: (2, 3)
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

// Test uint8 (Byte) data type - uses optimized xa_nn_transpose_8_8
TEST_F(HiFiTransposeCopyTest, UInt8Transpose) {
  TensorFactory<ScalarType::Byte> tf;

  // Input: (2, 3)
  const std::vector<int32_t> input_sizes{2, 3};
  const std::vector<int32_t> output_sizes{3, 2};

  std::vector<uint8_t> input_data = {1, 2, 3, 4, 5, 6};
  std::vector<uint8_t> expected_data = {1, 4, 2, 5, 3, 6};

  Tensor input = tf.make(input_sizes, input_data);
  Tensor out = tf.zeros(output_sizes);
  Tensor expected = tf.make(output_sizes, expected_data);

  transpose_copy_int_out(input, 0, 1, out);

  EXPECT_TENSOR_CLOSE(out, expected);
}

// Test with same dimensions (no-op case)
TEST_F(HiFiTransposeCopyTest, SameDimensionTranspose) {
  TensorFactory<ScalarType::Float> tf;

  // Input: (3, 3)
  const std::vector<int32_t> sizes{3, 3};

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  std::vector<float> expected_data = {1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f};

  Tensor input = tf.make(sizes, input_data);
  Tensor out = tf.zeros(sizes);
  Tensor expected = tf.make(sizes, expected_data);

  transpose_copy_int_out(input, 0, 1, out);

  EXPECT_TENSOR_CLOSE(out, expected);
}

} // namespace
} // namespace native
} // namespace HiFi
} // namespace impl
