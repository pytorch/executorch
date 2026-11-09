/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_sdpa_bitwise_mask_gen.h>

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

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::testing::TensorFactory;

// Transcription of the packing loop as it was written before the data pointers
// were hoisted out of it. Tests compare the kernel against this so that any
// rewrite of the loop body has to stay byte-identical.
template <typename T, typename MaskedPred>
std::vector<uint8_t> reference_pack(
    const std::vector<T>& in,
    MaskedPred is_masked) {
  std::vector<uint8_t> packed(in.size() / 8);
  for (size_t i = 0, out_index = 0; i < in.size(); i += 8, out_index++) {
    uint8_t packed_mask = 0;
    for (size_t j = 0; j < 8; j++) {
      packed_mask |= static_cast<uint8_t>(is_masked(in[i + j])) << j;
    }
    packed[out_index] = packed_mask;
  }
  return packed;
}

// A bool mask uses True = keep, so the kernel inverts it.
std::vector<uint8_t> reference_pack_bool(const std::vector<uint8_t>& in) {
  return reference_pack(in, [](uint8_t v) { return !v; });
}

std::vector<uint8_t> reference_pack_float(
    const std::vector<float>& in,
    double threshold) {
  return reference_pack(in, [threshold](float v) { return v < threshold; });
}

class GenericSdpaBitwiseMaskGenTest : public OperatorTest {
 protected:
  Tensor&
  sdpa_bitwise_mask_gen_out(const Tensor& mask, double threshold, Tensor& out) {
    return impl::generic::native::sdpa_bitwise_mask_gen_out(
        context_, mask, threshold, out);
  }
};

// Pins the bit order and polarity with hand-computed bytes: element j of each
// group of 8 lands in bit j, and a False (blocked) input sets that bit.
TEST_F(GenericSdpaBitwiseMaskGenTest, BoolMaskPacksBlockedAsSetBitLsbFirst) {
  TensorFactory<ScalarType::Bool> tf_bool;
  TensorFactory<ScalarType::Byte> tf_uint8;

  // clang-format off
  Tensor mask = tf_bool.make(
      {4, 8},
      {
          // All kept -> no bits set.
          1, 1, 1, 1, 1, 1, 1, 1,
          // All blocked -> every bit set.
          0, 0, 0, 0, 0, 0, 0, 0,
          // Only element 0 blocked -> bit 0.
          0, 1, 1, 1, 1, 1, 1, 1,
          // Only element 3 blocked -> bit 3.
          1, 1, 1, 0, 1, 1, 1, 1,
      });
  // clang-format on

  Tensor out = tf_uint8.zeros({4, 1});
  Tensor expected = tf_uint8.make({4, 1}, {0x00, 0xFF, 0x01, 0x08});

  sdpa_bitwise_mask_gen_out(mask, 0.0, out);

  EXPECT_TENSOR_EQ(out, expected);
}

// Values equal to the threshold are kept; only strictly smaller ones are
// masked.
TEST_F(GenericSdpaBitwiseMaskGenTest, FloatMaskThresholdIsStrictlyLess) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Byte> tf_uint8;

  const double threshold = -1.5;

  // clang-format off
  Tensor mask = tf_float.make(
      {3, 8},
      {
          // Exactly at the threshold, and above it -> kept.
          -1.5, -1.5, -1.5, -1.5, 0.0, 1.0, 100.0, -1.4999,
          // Below the threshold -> masked.
          -1.6, -2.0, -100.0, -1.50001, -3.0, -4.0, -5.0, -6.0,
          // Alternating, starting with a masked element.
          -2.0, 0.0, -2.0, 0.0, -2.0, 0.0, -2.0, 0.0,
      });
  // clang-format on

  Tensor out = tf_uint8.zeros({3, 1});
  Tensor expected = tf_uint8.make({3, 1}, {0x00, 0xFF, 0x55});

  sdpa_bitwise_mask_gen_out(mask, threshold, out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(GenericSdpaBitwiseMaskGenTest, BoolMaskMatchesReferencePacking) {
  TensorFactory<ScalarType::Bool> tf_bool;
  TensorFactory<ScalarType::Byte> tf_uint8;

  constexpr int kRows = 7;
  constexpr int kCols = 64;

  // Causal-style mask: row r keeps the first 8*(r+1) columns.
  std::vector<uint8_t> in(kRows * kCols);
  for (int r = 0; r < kRows; r++) {
    for (int c = 0; c < kCols; c++) {
      in[r * kCols + c] = static_cast<uint8_t>(c < 8 * (r + 1));
    }
  }

  Tensor mask = tf_bool.make({kRows, kCols}, in);
  Tensor out = tf_uint8.zeros({kRows, kCols / 8});
  Tensor expected = tf_uint8.make({kRows, kCols / 8}, reference_pack_bool(in));

  sdpa_bitwise_mask_gen_out(mask, 0.0, out);

  EXPECT_TENSOR_EQ(out, expected);
}

TEST_F(GenericSdpaBitwiseMaskGenTest, FloatMaskMatchesReferencePacking) {
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Byte> tf_uint8;

  constexpr int kRows = 7;
  constexpr int kCols = 64;
  const double threshold = 0.0;

  // Irregular sign pattern so every bit position gets exercised both ways.
  std::vector<float> in(kRows * kCols);
  for (size_t i = 0; i < in.size(); i++) {
    in[i] = static_cast<float>((i * 37) % 71) * 0.25f - 8.0f;
  }

  Tensor mask = tf_float.make({kRows, kCols}, in);
  Tensor out = tf_uint8.zeros({kRows, kCols / 8});
  Tensor expected =
      tf_uint8.make({kRows, kCols / 8}, reference_pack_float(in, threshold));

  sdpa_bitwise_mask_gen_out(mask, threshold, out);

  EXPECT_TENSOR_EQ(out, expected);
}

// A mask the size of a real attention mask, so the packing loop runs over many
// output bytes rather than a handful.
TEST_F(GenericSdpaBitwiseMaskGenTest, LargeMaskMatchesReferencePacking) {
  TensorFactory<ScalarType::Bool> tf_bool;
  TensorFactory<ScalarType::Float> tf_float;
  TensorFactory<ScalarType::Byte> tf_uint8;

  constexpr int kSeq = 256;
  constexpr int kKv = 256;
  const double threshold = 0.0;

  std::vector<uint8_t> bool_in(kSeq * kKv);
  std::vector<float> float_in(kSeq * kKv);
  for (int r = 0; r < kSeq; r++) {
    for (int c = 0; c < kKv; c++) {
      const bool keep = c <= r;
      bool_in[r * kKv + c] = static_cast<uint8_t>(keep);
      float_in[r * kKv + c] = keep ? 0.0f : -1e30f;
    }
  }

  Tensor bool_mask = tf_bool.make({kSeq, kKv}, bool_in);
  Tensor bool_out = tf_uint8.zeros({kSeq, kKv / 8});
  Tensor bool_expected =
      tf_uint8.make({kSeq, kKv / 8}, reference_pack_bool(bool_in));

  sdpa_bitwise_mask_gen_out(bool_mask, threshold, bool_out);

  EXPECT_TENSOR_EQ(bool_out, bool_expected);

  Tensor float_mask = tf_float.make({kSeq, kKv}, float_in);
  Tensor float_out = tf_uint8.zeros({kSeq, kKv / 8});
  Tensor float_expected =
      tf_uint8.make({kSeq, kKv / 8}, reference_pack_float(float_in, threshold));

  sdpa_bitwise_mask_gen_out(float_mask, threshold, float_out);

  EXPECT_TENSOR_EQ(float_out, float_expected);

  // Both dtypes describe the same mask, so they must pack to the same bytes.
  EXPECT_TENSOR_EQ(bool_out, float_out);
}

} // namespace
} // namespace native
} // namespace generic
} // namespace impl
