/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/quantized_linear_packed.h>

#include <executorch/backends/cadence/generic/operators/op_quantized_fully_connected.h>
#include <executorch/backends/cadence/generic/operators/op_quantized_fully_connected_packed.h>
#include <executorch/backends/cadence/generic/operators/quantized_linear.h>
#include <executorch/kernels/test/TestUtil.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_factory.h>
#include <executorch/runtime/core/exec_aten/testing_util/tensor_util.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace impl {
namespace generic {
namespace native {
namespace {

using ::executorch::aten::Tensor;
using ::executorch::runtime::testing::TensorFactory;

// Mirror of the AoT packer in
// executorch/backends/cadence/aot/weight_packing.py. Kept independent of the
// kernel's decode path on purpose: the test is only meaningful if the two
// sides are written separately and still agree.
std::vector<uint8_t> pack6(
    const std::vector<int8_t>& values,
    const size_t in_dim) {
  // Planar 4+2, per row:
  //   plane A (in_dim/2 B): byte k = low nibbles of values k, k+in_dim/2
  //   plane B (in_dim/4 B): byte m = 2-bit fields of m, m+Q, m+2Q, m+3Q
  // Pairing k with k+in_dim/2 is what makes a decoder emit natural k order.
  EXPECT_EQ(in_dim % 4, 0u);
  EXPECT_EQ(values.size() % in_dim, 0u);
  const size_t half = in_dim / 2, quarter = in_dim / 4;
  std::vector<uint8_t> out;
  for (size_t base = 0; base < values.size(); base += in_dim) {
    for (size_t k = 0; k < half; ++k) {
      const uint8_t lo = static_cast<uint8_t>(values[base + k] + 32) & 0x0F;
      const uint8_t hi =
          static_cast<uint8_t>(values[base + k + half] + 32) & 0x0F;
      out.push_back(lo | (hi << 4));
    }
    for (size_t m = 0; m < quarter; ++m) {
      uint8_t b = 0;
      for (size_t q = 0; q < 4; ++q) {
        const uint8_t f =
            (static_cast<uint8_t>(values[base + m + q * quarter] + 32) >> 4) &
            0x03;
        b |= static_cast<uint8_t>(f << (2 * q));
      }
      out.push_back(b);
    }
  }
  return out;
}

// Split 4-bit, per row: byte k = value k in the low nibble, value k+in_dim/2
// in the high nibble. Same pairing as plane A above, and for the same reason.
// Deliberately written independently of the kernel decode so agreement means
// something.
std::vector<uint8_t> pack4(
    const std::vector<int8_t>& values,
    const size_t in_dim) {
  EXPECT_EQ(in_dim % 2, 0u);
  EXPECT_EQ(values.size() % in_dim, 0u);
  const size_t half = in_dim / 2;
  std::vector<uint8_t> out;
  for (size_t base = 0; base < values.size(); base += in_dim) {
    for (size_t k = 0; k < half; ++k) {
      const uint8_t lo = static_cast<uint8_t>(values[base + k] + 8) & 0x0F;
      const uint8_t hi =
          static_cast<uint8_t>(values[base + k + half] + 8) & 0x0F;
      out.push_back(lo | (hi << 4));
    }
  }
  return out;
}

std::vector<int8_t> as_int8(const std::vector<uint8_t>& bytes) {
  return std::vector<int8_t>(bytes.begin(), bytes.end());
}

class QuantizedLinearPackedTest : public OperatorTest {};

// A packed weight must produce exactly the same output as the dense weight it
// was packed from. This is the property an optimized HiFi kernel has to keep.
TEST_F(QuantizedLinearPackedTest, MatchesDenseKernelSixBit) {
  constexpr int64_t kOutDim = 3;
  constexpr int64_t kInDim = 8;

  // spans the full 6-bit range including both endpoints
  const std::vector<int8_t> dense = {
      -32, 31, 0,  -1,  1,  17,  -17, 30, //
      5,   -5, 12, -12, 31, -32, 2,   -2, //
      0,   0,  0,  0,   -1, 1,   -1,  1,
  };
  const std::vector<int8_t> packed = as_int8(pack6(dense, kInDim));
  ASSERT_EQ(packed.size(), static_cast<size_t>(kOutDim * kInDim * 6 / 8));

  TensorFactory<::executorch::aten::ScalarType::Char> tf_char;
  TensorFactory<::executorch::aten::ScalarType::Int> tf_int;

  Tensor src = tf_char.make({1, kInDim}, {7, -3, 0, 12, -40, 5, 60, -1});
  Tensor dense_w = tf_char.make({kOutDim, kInDim}, dense);
  Tensor packed_w = tf_char.make(
      {static_cast<int32_t>(kOutDim),
       static_cast<int32_t>(packed.size() / kOutDim)},
      packed);
  Tensor bias = tf_int.make({kOutDim}, {10, -20, 30});
  Tensor out_multiplier = tf_int.make({kOutDim}, {1 << 30, 1 << 29, 1 << 30});
  Tensor out_shift = tf_int.make({kOutDim}, {0, 1, -1});

  Tensor expected = tf_char.zeros({1, kOutDim});
  Tensor actual = tf_char.zeros({1, kOutDim});

  ::impl::generic::quantized::quantized_linear_per_channel_<int8_t>(
      src,
      dense_w,
      bias,
      /*src_zero_point=*/-5,
      /*weight_zero_point=*/0,
      out_multiplier,
      out_shift,
      /*out_zero_point=*/2,
      expected);

  ::impl::generic::quantized::quantized_linear_packed_per_channel_<int8_t>(
      src,
      packed_w,
      bias,
      kInDim,
      /*weight_bits=*/6,
      /*src_zero_point=*/-5,
      /*weight_zero_point=*/0,
      out_multiplier,
      out_shift,
      /*out_zero_point=*/2,
      actual);

  EXPECT_TENSOR_EQ(actual, expected);
}

TEST_F(QuantizedLinearPackedTest, MatchesDenseKernelFourBit) {
  constexpr int64_t kOutDim = 2;
  constexpr int64_t kInDim = 6;

  const std::vector<int8_t> dense = {
      -8,
      7,
      0,
      -1,
      1,
      3, //
      2,
      -2,
      5,
      -5,
      7,
      -8,
  };
  const std::vector<int8_t> packed = as_int8(pack4(dense, kInDim));

  TensorFactory<::executorch::aten::ScalarType::Char> tf_char;
  TensorFactory<::executorch::aten::ScalarType::Int> tf_int;

  Tensor src = tf_char.make({1, kInDim}, {1, -2, 3, -4, 5, -6});
  Tensor dense_w = tf_char.make({kOutDim, kInDim}, dense);
  Tensor packed_w = tf_char.make(
      {static_cast<int32_t>(kOutDim),
       static_cast<int32_t>(packed.size() / kOutDim)},
      packed);
  Tensor bias = tf_int.make({kOutDim}, {0, 4});
  Tensor out_multiplier = tf_int.make({kOutDim}, {1 << 30, 1 << 30});
  Tensor out_shift = tf_int.make({kOutDim}, {1, 0});

  Tensor expected = tf_char.zeros({1, kOutDim});
  Tensor actual = tf_char.zeros({1, kOutDim});

  ::impl::generic::quantized::quantized_linear_per_channel_<int8_t>(
      src, dense_w, bias, 0, 0, out_multiplier, out_shift, 0, expected);

  ::impl::generic::quantized::quantized_linear_packed_per_channel_<int8_t>(
      src,
      packed_w,
      bias,
      kInDim,
      /*weight_bits=*/4,
      0,
      0,
      out_multiplier,
      out_shift,
      0,
      actual);

  EXPECT_TENSOR_EQ(actual, expected);
}

// out_dim has to survive as shape[0] of the packed tensor: shape inference and
// the fully-connected meta kernels read it.
TEST_F(QuantizedLinearPackedTest, PreservesOutDimAndBatches) {
  constexpr int64_t kOutDim = 2;
  constexpr int64_t kInDim = 4;

  const std::vector<int8_t> dense = {1, -1, 2, -2, 3, -3, 4, -4};
  const std::vector<int8_t> packed = as_int8(pack6(dense, kInDim));

  TensorFactory<::executorch::aten::ScalarType::Char> tf_char;
  TensorFactory<::executorch::aten::ScalarType::Int> tf_int;

  Tensor packed_w = tf_char.make({kOutDim, 3}, packed);
  EXPECT_EQ(packed_w.size(0), kOutDim);

  // two rows of activations exercise the leading_dims loop
  Tensor src = tf_char.make({2, kInDim}, {1, 2, 3, 4, -1, -2, -3, -4});
  Tensor dense_w = tf_char.make({kOutDim, kInDim}, dense);
  Tensor bias = tf_int.make({kOutDim}, {0, 0});
  Tensor out_multiplier = tf_int.make({kOutDim}, {1 << 30, 1 << 30});
  Tensor out_shift = tf_int.make({kOutDim}, {0, 0});

  Tensor expected = tf_char.zeros({2, kOutDim});
  Tensor actual = tf_char.zeros({2, kOutDim});

  ::impl::generic::quantized::quantized_linear_per_channel_<int8_t>(
      src, dense_w, bias, 0, 0, out_multiplier, out_shift, 0, expected);

  ::impl::generic::quantized::quantized_linear_packed_per_channel_<int8_t>(
      src,
      packed_w,
      bias,
      kInDim,
      6,
      0,
      0,
      out_multiplier,
      out_shift,
      0,
      actual);

  EXPECT_TENSOR_EQ(actual, expected);
}

// The registered operator, not just the inner kernel: a packed weight through
// quantized_fully_connected_packed_out must match the dense weight through
// quantized_fully_connected_out. This is what the runtime actually dispatches.
TEST_F(QuantizedLinearPackedTest, OpMatchesDenseFullyConnected) {
  constexpr int64_t kOutDim = 2;
  constexpr int64_t kInDim = 8;

  const std::vector<int8_t> dense = {
      -32,
      31,
      0,
      -1,
      1,
      17,
      -17,
      30, //
      5,
      -5,
      12,
      -12,
      31,
      -32,
      2,
      -2,
  };
  const std::vector<int8_t> packed = as_int8(pack6(dense, kInDim));

  TensorFactory<::executorch::aten::ScalarType::Char> tf_char;
  TensorFactory<::executorch::aten::ScalarType::Int> tf_int;

  Tensor src = tf_char.make({1, kInDim}, {7, -3, 0, 12, -40, 5, 60, -1});
  Tensor dense_w = tf_char.make({kOutDim, kInDim}, dense);
  Tensor packed_w = tf_char.make(
      {static_cast<int32_t>(kOutDim),
       static_cast<int32_t>(packed.size() / kOutDim)},
      packed);
  Tensor bias = tf_int.make({kOutDim}, {10, -20});
  Tensor weight_zero_point = tf_int.make({1}, {0});
  Tensor out_multiplier = tf_int.make({kOutDim}, {1 << 30, 1 << 29});
  Tensor out_shift = tf_int.make({kOutDim}, {0, 1});

  Tensor expected = tf_char.zeros({1, kOutDim});
  Tensor actual = tf_char.zeros({1, kOutDim});

  quantized_fully_connected_out(
      context_,
      src,
      dense_w,
      bias,
      /*in_zero_point=*/-5,
      weight_zero_point,
      out_multiplier,
      out_shift,
      /*out_zero_point=*/2,
      std::nullopt,
      expected);

  quantized_fully_connected_packed_out(
      context_,
      src,
      packed_w,
      bias,
      kInDim,
      /*weight_bits=*/6,
      /*in_zero_point=*/-5,
      weight_zero_point,
      out_multiplier,
      out_shift,
      /*out_zero_point=*/2,
      std::nullopt,
      actual);

  EXPECT_TENSOR_EQ(actual, expected);
}

// A per-tensor source lifts its scalar qparams to length-1 tensors while the
// output still has many channels, so the kernel must stop indexing them by
// channel. Every layer of a real model takes this path.
TEST_F(
    QuantizedLinearPackedTest,
    OpMatchesDenseFullyConnectedPerTensorQParams) {
  constexpr int64_t kOutDim = 2;
  constexpr int64_t kInDim = 8;

  const std::vector<int8_t> dense = {
      -32,
      31,
      0,
      -1,
      1,
      17,
      -17,
      30, //
      5,
      -5,
      12,
      -12,
      31,
      -32,
      2,
      -2,
  };
  const std::vector<int8_t> packed = as_int8(pack6(dense, kInDim));

  TensorFactory<::executorch::aten::ScalarType::Char> tf_char;
  TensorFactory<::executorch::aten::ScalarType::Int> tf_int;

  Tensor src = tf_char.make({1, kInDim}, {7, -3, 0, 12, -40, 5, 60, -1});
  Tensor dense_w = tf_char.make({kOutDim, kInDim}, dense);
  Tensor packed_w = tf_char.make(
      {static_cast<int32_t>(kOutDim),
       static_cast<int32_t>(packed.size() / kOutDim)},
      packed);
  Tensor bias = tf_int.make({kOutDim}, {10, -20});
  Tensor weight_zero_point = tf_int.make({1}, {0});
  Tensor out_multiplier = tf_int.make({1}, {1 << 30});
  Tensor out_shift = tf_int.make({1}, {0});

  Tensor expected = tf_char.zeros({1, kOutDim});
  Tensor actual = tf_char.zeros({1, kOutDim});

  quantized_fully_connected_out(
      context_,
      src,
      dense_w,
      bias,
      /*in_zero_point=*/-5,
      weight_zero_point,
      out_multiplier,
      out_shift,
      /*out_zero_point=*/2,
      std::nullopt,
      expected);

  quantized_fully_connected_packed_out(
      context_,
      src,
      packed_w,
      bias,
      kInDim,
      /*weight_bits=*/6,
      /*in_zero_point=*/-5,
      weight_zero_point,
      out_multiplier,
      out_shift,
      /*out_zero_point=*/2,
      std::nullopt,
      actual);

  EXPECT_TENSOR_EQ(actual, expected);
}

} // namespace
} // namespace native
} // namespace generic
} // namespace impl
